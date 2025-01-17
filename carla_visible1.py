#!/usr/bin/env python

import gym
import numpy as np
import random
import math
import carla
import pygame
import sys
import cv2

from gym import spaces
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MultiInputLstmPolicy

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# ----------------------------------------------------------------------------------
# Color palette for semantic segmentation in CARLA
# (Partial example; fill out more classes as needed)
# E.g. class '7' is road -> (128,64,128)
# You can find full classes in the CARLA docs or do your own mapping
# ----------------------------------------------------------------------------------
SEMSEG_PALETTE = {
    0:  (0, 0, 0),          # Unlabeled
    1:  (70, 70, 70),       # Building
    2:  (190, 153, 153),    # Fence
    3:  (72, 0, 90),        # Other
    4:  (220, 20, 60),      # Pedestrian
    5:  (153, 153, 153),    # Pole
    6:  (157, 234, 50),     # RoadLines
    7:  (128, 64, 128),     # Road
    8:  (244, 35, 232),     # Sidewalk
    9:  (107, 142, 35),     # Vegetation
    10: (0, 0, 142),        # Vehicle
    11: (102, 102, 156),    # Wall
    12: (220, 220, 0),      # TrafficSign
    # ... add more if needed for your CARLA version
}


def apply_semseg_palette(semseg_id_img: np.ndarray) -> np.ndarray:
    """
    Map each pixel's class ID to a color using SEMSEG_PALETTE.
    Input semseg_id_img: shape (H, W), each pixel in [0..22 or so].
    Output: color-coded image shape (H, W, 3).
    """
    height, width = semseg_id_img.shape[:2]
    colored_img = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id, color in SEMSEG_PALETTE.items():
        colored_img[semseg_id_img == class_id] = color
    return colored_img


# ---------------------------------------------------------------------
# Helper: Convert a 3D world point to camera pixel coordinates (approx)
# ---------------------------------------------------------------------
def world_to_camera(world_loc, camera_transform, camera_intrinsic, world):
    """
    Approximate conversion of a 3D world location (carla.Location) to the
    pixel coordinates in the camera image plane.
    """
    world_point = np.array([world_loc.x, world_loc.y, world_loc.z, 1.0])

    # Build the matrix from camera_transform (camera->world).
    c_yaw = math.radians(camera_transform.rotation.yaw)
    c_pitch = math.radians(camera_transform.rotation.pitch)
    c_roll = math.radians(camera_transform.rotation.roll)

    R_yaw = np.array([[ math.cos(c_yaw), -math.sin(c_yaw), 0],
                      [ math.sin(c_yaw),  math.cos(c_yaw), 0],
                      [             0,               0,    1]])
    R_pitch = np.array([[ math.cos(c_pitch), 0, math.sin(c_pitch)],
                        [                 0, 1,                0],
                        [-math.sin(c_pitch),0, math.cos(c_pitch)]])
    R_roll = np.array([[1,             0,              0],
                       [0, math.cos(c_roll), -math.sin(c_roll)],
                       [0, math.sin(c_roll),  math.cos(c_roll)]])
    R = R_roll @ R_pitch @ R_yaw

    t = np.array([camera_transform.location.x,
                  camera_transform.location.y,
                  camera_transform.location.z])

    # world->camera = inverse(camera->world)
    R_inv = R.T
    t_inv = -R_inv @ t

    transform_mat = np.eye(4)
    transform_mat[0:3,0:3] = R_inv
    transform_mat[0:3,3] = t_inv

    cam_coords = transform_mat @ world_point  # shape (4,)

    # If behind camera
    if cam_coords[2] <= 0.01:
        return None

    # Project to image plane
    px = (camera_intrinsic[0, 0] * (cam_coords[0] / cam_coords[2])) + camera_intrinsic[0, 2]
    py = (camera_intrinsic[1, 1] * (cam_coords[1] / cam_coords[2])) + camera_intrinsic[1, 2]

    return (int(px), int(py))


# ----------------------------------------------------------------------------------
# CARLA Environment: Multi-input (image + vector). Now with a color palette fix.
# ----------------------------------------------------------------------------------
class CarlaMultiInputEnv(gym.Env):
    """
    - Observation: Dict{"image": (3,H,W), "vector": shape(7,)}
    - Action: [steer, throttle], with ranges [-0.5,0.5] x [0.0,0.5]
    - Episode length: 2500 steps
    - Render with PyGame: show color-coded seg camera + text
    - Fix for black image: apply semantic palette
    """

    def __init__(self, config):
        super(CarlaMultiInputEnv, self).__init__()

        self.host = config.get("host", "localhost")
        self.port = config.get("port", 2000)
        self.timeout = config.get("timeout", 10.0)
        self.synchronous_mode = config.get("synchronous_mode", True)

        self.img_width = config.get("img_width", 84)
        self.img_height = config.get("img_height", 84)
        self.max_episode_steps = 2500
        self.episode_step = 0

        # Connect to CARLA
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()
        self.map = self.world.get_map()

        if self.synchronous_mode:
            settings = self.world.get_settings()
            settings.fixed_delta_seconds = 0.05
            settings.synchronous_mode = True
            self.world.apply_settings(settings)

        self.bp_library = self.world.get_blueprint_library()

        # Action space: 2D [steer, throttle]
        self.action_space = spaces.Box(
            low=np.array([-0.5, 0.0], dtype=np.float32),
            high=np.array([0.5, 0.5], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        # Dict observation
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=255,
                shape=(3, self.img_height, self.img_width),
                dtype=np.uint8
            ),
            "vector": spaces.Box(
                low=-1e5, high=1e5,
                shape=(7,),
                dtype=np.float32
            )
        })

        self.vehicle = None
        self.collision_sensor = None
        self.camera_sensor = None
        self.collision_hist = []
        self.front_image = np.zeros((3, self.img_height, self.img_width), dtype=np.uint8)

        self.use_render = config.get("use_render", True)
        self.screen = None
        self.clock = None
        self.font = None

        self.camera_transform = None
        self.intrinsic = self._build_camera_intrinsic()

    def _build_camera_intrinsic(self):
        f_x = self.img_width / (2.0 * math.tan(math.radians(90.0)/2.0))
        f_y = f_x
        c_x = self.img_width/2.0
        c_y = self.img_height/2.0
        return np.array([
            [f_x,   0,     c_x],
            [0,     f_y,   c_y],
            [0,     0,     1]
        ])

    def reset(self):
        self._destroy_actors()
        self.collision_hist.clear()
        self.episode_step = 0

        # Spawn vehicle
        spawn_points = self.map.get_spawn_points()
        if not spawn_points:
            raise ValueError("No spawn points in CARLA map!")
        spawn_point = random.choice(spawn_points)

        vehicle_bp = self.bp_library.find('vehicle.tesla.model3')
        self.vehicle = None
        for _ in range(10):
            actor = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if actor is not None:
                self.vehicle = actor
                break
        if self.vehicle is None:
            raise ValueError("Could not spawn vehicle after multiple tries.")

        # Collision sensor
        col_bp = self.bp_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda e: self._on_collision(e))

        # Semantic seg camera
        cam_bp = self.bp_library.find('sensor.camera.semantic_segmentation')
        cam_bp.set_attribute("image_size_x", f"{self.img_width}")
        cam_bp.set_attribute("image_size_y", f"{self.img_height}")
        cam_bp.set_attribute("fov", "90")
        cam_transform = carla.Transform(carla.Location(x=1.6, z=2.0))
        self.camera_sensor = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.vehicle)
        self.camera_sensor.listen(lambda img: self._process_camera(img))

        self.camera_transform = self.camera_sensor.get_transform()

        # Stabilize
        for _ in range(20):
            if self.synchronous_mode:
                self.world.tick()
            else:
                self.world.wait_for_tick()

        if self.use_render and self.screen is None:
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((800, 600))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 20)

        return self._get_obs()

    def step(self, action):
        self.episode_step += 1

        steer, throttle = action
        control = carla.VehicleControl(
            steer=float(steer),
            throttle=float(throttle),
            brake=0.0
        )
        self.vehicle.apply_control(control)

        if self.synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        obs = self._get_obs()
        reward, done = self._compute_reward_done()

        return obs, reward, done, {}

    # --------------------- The main fix is in _process_camera ----------------------
    def _process_camera(self, image):
        """
        Convert CARLA semantic seg raw_data to color-coded image => (3,H,W).
        Without color-mapping, it may appear black if the class IDs are small.
        """
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        sem_id = array[:, :, 2]  # channel 2 is the class ID in default BGRA layout

        # Apply color palette so we get a visible colored image
        seg_colored = apply_semseg_palette(sem_id)  # shape (H,W,3)

        # Transpose to (3,H,W) for SB3
        seg_colored_t = np.transpose(seg_colored, (2, 0, 1))
        self.front_image = seg_colored_t

        if self.camera_sensor:
            self.camera_transform = self.camera_sensor.get_transform()

    # -----------------------------------------------------------------
    def render(self, mode='human'):
        if not self.use_render or self.screen is None:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Convert (3,H,W)->(H,W,3)
        img = np.transpose(self.front_image, (1,2,0))
        surf = pygame.surfarray.make_surface(img)
        camera_surf = pygame.transform.scale(surf, (400, 400))

        self.screen.fill((0,0,0))
        self.screen.blit(camera_surf, (0,0))

        # Draw text
        vec_obs = self._get_vector_obs()
        nx, ny, cx, cy, speed, lane_off, heading_err = vec_obs
        lines = [
            f"Step: {self.episode_step}/{self.max_episode_steps}",
            f"Speed: {speed:.2f} m/s",
            f"Lane Offset: {lane_off:.2f}",
            f"Heading Err: {heading_err:.2f}",
            f"Cur Pos: ({cx:.1f}, {cy:.1f})",
            f"Next WP: ({nx:.1f}, {ny:.1f})",
        ]
        x_text, y_text = 420, 10
        for line in lines:
            text_surf = self.font.render(line, True, (255,255,255))
            self.screen.blit(text_surf, (x_text, y_text))
            y_text += 25

        # Attempt dot for next WP
        waypoint_loc = carla.Location(x=nx, y=ny, z=0.0)
        dot_coords = world_to_camera(waypoint_loc, self.camera_transform, self.intrinsic, self.world)
        if dot_coords:
            px, py = dot_coords
            scale_x = 400 / self.img_width
            scale_y = 400 / self.img_height
            draw_x = int(px * scale_x)
            draw_y = int(py * scale_y)
            if 0 <= draw_x < 400 and 0 <= draw_y < 400:
                pygame.draw.circle(camera_surf, (255,0,0), (draw_x, draw_y), 5)
        self.screen.blit(camera_surf, (0,0))

        pygame.display.flip()
        self.clock.tick(30)

    def _compute_reward_done(self):
        done = False
        if len(self.collision_hist) > 0:
            return -30.0, True
        if self.episode_step >= self.max_episode_steps:
            done = True

        vec_obs = self._get_vector_obs()
        _, _, _, _, speed, lane_off, heading_err = vec_obs
        target_speed = 10.0
        heading_pen = -0.1 * abs(heading_err)
        offset_pen = -0.05 * abs(lane_off)
        speed_pen = -0.01 * abs(speed - target_speed)
        step_reward = heading_pen + offset_pen + speed_pen
        step_reward = float(np.clip(step_reward, -1.0, 1.0))
        return step_reward, done

    def _get_obs(self):
        return {
            "image": self.front_image,
            "vector": self._get_vector_obs()
        }

    def _get_vector_obs(self):
        tf = self.vehicle.get_transform()
        vx, vy = tf.location.x, tf.location.y
        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        lane_off, heading_err = self._lane_center_offset_and_heading(tf)

        # Next WP ~5m ahead
        waypoint = self.map.get_waypoint(tf.location, project_to_road=True)
        if waypoint:
            wps = waypoint.next(5.0)
            if len(wps)>0:
                nx, ny = wps[0].transform.location.x, wps[0].transform.location.y
            else:
                nx, ny = vx, vy
        else:
            nx, ny = vx, vy

        return np.array([nx, ny, vx, vy, speed, lane_off, heading_err], dtype=np.float32)

    def _lane_center_offset_and_heading(self, vehicle_transform):
        wp = self.map.get_waypoint(
            vehicle_transform.location, project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if not wp:
            return 0.0, 0.0

        lane_tf = wp.transform
        vx, vy = vehicle_transform.location.x, vehicle_transform.location.y
        vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)

        lx, ly = lane_tf.location.x, lane_tf.location.y
        lane_yaw = math.radians(lane_tf.rotation.yaw)

        right_vec = lane_tf.get_right_vector()
        dx, dy = (vx - lx), (vy - ly)
        lane_offset = dx * right_vec.x + dy * right_vec.y
        heading_err = vehicle_yaw - lane_yaw
        heading_err = (heading_err + math.pi) % (2*math.pi) - math.pi
        return lane_offset, heading_err

    def _destroy_actors(self):
        if self.collision_sensor:
            self.collision_sensor.destroy()
            self.collision_sensor = None
        if self.camera_sensor:
            self.camera_sensor.destroy()
            self.camera_sensor = None
        if self.vehicle:
            self.vehicle.destroy()
            self.vehicle = None

    def close(self):
        self._destroy_actors()
        if self.synchronous_mode:
            s = self.world.get_settings()
            s.synchronous_mode = False
            s.fixed_delta_seconds = None
            self.world.apply_settings(s)
        if self.screen:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None


def main():
    config = {
        "host": "localhost",
        "port": 2000,
        "timeout": 10.0,
        "synchronous_mode": True,
        "img_width": 84,
        "img_height": 84,
        "use_render": True   # We'll show PyGame
    }

    def make_env():
        return CarlaMultiInputEnv(config)

    env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs_multiinput_lstm/",
        log_path="./logs_multiinput_lstm/",
        eval_freq=50_000,
        deterministic=True,
        render=False
    )

    model = RecurrentPPO(
        policy=MultiInputLstmPolicy,
        env=env,
        n_steps=128,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        n_epochs=10,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./multiinput_lstm_tensorboard/",
        policy_kwargs=dict(
            n_lstm_layers=2,
            lstm_hidden_size=512,
            net_arch=[dict(pi=[512,512,512], vf=[512,512,512])],
            cnn_extractor_kwargs=dict(features_dim=512),
        )
    )

    model.learn(
        total_timesteps=200000,  # or more
        callback=eval_callback
    )
    model.save("multiinput_cnn_lstm_carla_pygame")
    print("Training complete! Model saved.")


if __name__ == "__main__":
    main()
