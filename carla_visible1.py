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


# ---------------------------------------------------------------------
# Helper: Convert a 3D world point to camera pixel coordinates (approx)
# ---------------------------------------------------------------------
def world_to_camera(world_loc, camera_transform, camera_intrinsic, world):
    """
    Approximate conversion of a 3D world location (carla.Location) to the
    pixel coordinates in the camera image plane.

    camera_transform is a carla.Transform of the camera in world coords.
    camera_intrinsic is a 3x3 numpy array for the pinhole camera model.
    Return (u, v) in pixel coords, or None if behind camera / out of view.
    """

    # 1) Transform the 3D point from world coords -> camera coords
    #    We do a simplified approach: first transform world->camera
    #    using the inverse of camera_transform
    #    For a real pipeline, you'd convert from world -> vehicle -> camera.
    #    This snippet uses just camera -> world if the camera is attached to the vehicle with no extra transforms.
    world_point = np.array([world_loc.x, world_loc.y, world_loc.z, 1.0])
    
    # Build matrix from camera_transform
    # Rotation (roll, pitch, yaw) => ignoring roll for typical cameras. We'll do a standard approach:
    c_yaw = math.radians(camera_transform.rotation.yaw)
    c_pitch = math.radians(camera_transform.rotation.pitch)
    c_roll = math.radians(camera_transform.rotation.roll)

    # Construct rotation matrix (RPY) in a simplistic way
    # (This is a standard approach but can differ from CARLA's if not carefully matched.)
    R_yaw = np.array([[ math.cos(c_yaw), -math.sin(c_yaw), 0],
                      [ math.sin(c_yaw),  math.cos(c_yaw), 0],
                      [             0,               0,    1]])
    R_pitch = np.array([[ math.cos(c_pitch), 0, math.sin(c_pitch)],
                        [                 0, 1,                 0],
                        [-math.sin(c_pitch),0, math.cos(c_pitch)]])
    R_roll = np.array([[1,              0,               0],
                       [0, math.cos(c_roll), -math.sin(c_roll)],
                       [0, math.sin(c_roll),  math.cos(c_roll)]])
    R = R_roll @ R_pitch @ R_yaw
    
    # Translation
    t = np.array([camera_transform.location.x,
                  camera_transform.location.y,
                  camera_transform.location.z])

    # Build 4x4 transform world->camera
    # Actually we want the inverse of camera->world
    # camera->world = T * R; so world->camera = (camera->world)^(-1)
    # For simplicity, let's do it manually:
    R_inv = R.T
    t_inv = -R_inv @ t

    transform_mat = np.eye(4)
    transform_mat[0:3,0:3] = R_inv
    transform_mat[0:3,3] = t_inv

    cam_coords = transform_mat @ world_point  # shape (4,)

    # If the point is behind the camera (cam_coords.z <= 0), discard
    if cam_coords[2] <= 0.01:
        return None

    # 2) Project to image using camera_intrinsic
    px = (camera_intrinsic[0, 0] * (cam_coords[0] / cam_coords[2])) + camera_intrinsic[0, 2]
    py = (camera_intrinsic[1, 1] * (cam_coords[1] / cam_coords[2])) + camera_intrinsic[1, 2]

    return (int(px), int(py))


# ----------------------------------------------------------------------------------
# CARLA Environment: Multi-input (image + vector). Now with PyGame render().
# ----------------------------------------------------------------------------------
class CarlaMultiInputEnv(gym.Env):
    """
    - Observation: Dict{"image": (3,H,W), "vector": shape(7,)}
    - Action: [steer, throttle], with ranges [-0.5,0.5] x [0.0,0.5]
    - Episode length: 2500 steps
    - Render: shows segmentation camera + text info, plus a dot for next waypoint
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
            settings.fixed_delta_seconds = 0.05  # ~20 FPS
            settings.synchronous_mode = True
            self.world.apply_settings(settings)

        self.bp_library = self.world.get_blueprint_library()

        # Action space
        self.action_space = spaces.Box(
            low=np.array([-0.5,  0.0], dtype=np.float32),
            high=np.array([ 0.5, 0.5], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        # Observation space: Dict with "image" and "vector"
        # "image": (3,H,W) in [0..255], "vector": shape(7,)
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

        # Actors / sensors
        self.vehicle = None
        self.collision_sensor = None
        self.camera_sensor = None
        self.collision_hist = []

        # Store latest front image as (3,H,W)
        self.front_image = np.zeros((3, self.img_height, self.img_width), dtype=np.uint8)

        # PyGame for rendering
        self.use_render = config.get("use_render", True)
        self.screen = None
        self.clock = None
        self.font = None

        # We'll store the camera transform once spawned
        self.camera_transform = None
        # We'll build an approximate camera intrinsic
        self.intrinsic = self._build_camera_intrinsic()

    def _build_camera_intrinsic(self):
        """
        Build a basic pinhole camera intrinsic for the segmentation camera.
        We'll ignore any lens distortion, etc.
        """
        f_x = self.img_width / (2.0 * math.tan(math.radians(90.0) / 2.0))  # from fov=90
        f_y = f_x
        c_x = self.img_width / 2.0
        c_y = self.img_height / 2.0

        K = np.array([
            [f_x,    0.0,  c_x],
            [ 0.0,   f_y,  c_y],
            [ 0.0,   0.0,  1.0]
        ])
        return K

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
        self.collision_sensor = self.world.spawn_actor(
            col_bp, carla.Transform(), attach_to=self.vehicle
        )
        self.collision_sensor.listen(lambda e: self._on_collision(e))

        # Camera sensor (semantic seg)
        cam_bp = self.bp_library.find('sensor.camera.semantic_segmentation')
        cam_bp.set_attribute("image_size_x", f"{self.img_width}")
        cam_bp.set_attribute("image_size_y", f"{self.img_height}")
        cam_bp.set_attribute("fov", "90")
        # Attach in front of the vehicle
        cam_transform = carla.Transform(carla.Location(x=1.6, z=2.0))
        self.camera_sensor = self.world.spawn_actor(
            cam_bp, cam_transform, attach_to=self.vehicle
        )
        self.camera_sensor.listen(lambda img: self._process_camera(img))

        # We'll store the camera transform in world coords
        self.camera_transform = self.camera_sensor.get_transform()

        # Wait to stabilize
        for _ in range(20):
            if self.synchronous_mode:
                self.world.tick()
            else:
                self.world.wait_for_tick()

        # Pygame init if we plan to render
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

    def render(self, mode='human'):
        """
        PyGame-based rendering:
          1) Show the segmentation camera on the left side of the window.
          2) Overlay scalar info (speed, offset, heading, etc.).
          3) Draw a small dot for the next waypoint if it is in the camera FOV.
        """
        if not self.use_render or self.screen is None:
            return  # no rendering

        # Pump PyGame events to avoid freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Convert self.front_image (3,H,W) -> (H,W,3)
        # Then convert to a PyGame surface
        img = np.transpose(self.front_image, (1, 2, 0))  # shape (H,W,3)
        surf = pygame.surfarray.make_surface(img)

        # Scale the camera view to something bigger, e.g., (400,400)
        camera_surf = pygame.transform.scale(surf, (400, 400))

        # Fill background
        self.screen.fill((0,0,0))
        # Blit the camera feed at (0,0)
        self.screen.blit(camera_surf, (0, 0))

        # Draw text for the scalar states
        vec_obs = self._get_vector_obs()
        next_wp_x, next_wp_y, curr_x, curr_y, speed, lane_off, heading_err = vec_obs
        lines = [
            f"Step: {self.episode_step}/{self.max_episode_steps}",
            f"Speed: {speed:.2f} m/s",
            f"Lane Offset: {lane_off:.2f}",
            f"Heading Err: {heading_err:.2f}",
            f"Current Pos: ({curr_x:.1f}, {curr_y:.1f})",
            f"Next WP: ({next_wp_x:.1f}, {next_wp_y:.1f})"
        ]
        x_text, y_text = 420, 10
        for line in lines:
            text_surf = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(text_surf, (x_text, y_text))
            y_text += 25

        # Attempt to draw the next waypoint as a dot in the camera feed
        waypoint_loc = carla.Location(x=next_wp_x, y=next_wp_y, z=0.0)
        dot_coords = world_to_camera(waypoint_loc, self.camera_transform, self.intrinsic, self.world)
        if dot_coords is not None:
            dot_x, dot_y = dot_coords
            # The camera feed was scaled to (400,400), so we scale dot coords
            scale_x = 400 / self.img_width
            scale_y = 400 / self.img_height
            draw_x = int(dot_x * scale_x)
            draw_y = int(dot_y * scale_y)

            # Check if on-screen
            if 0 <= draw_x < 400 and 0 <= draw_y < 400:
                # Draw small circle on top of camera feed
                pygame.draw.circle(camera_surf, (255, 0, 0), (draw_x, draw_y), 5)

        self.screen.blit(camera_surf, (0, 0))

        pygame.display.flip()
        self.clock.tick(30)  # limit to ~30 FPS in UI

    # ---------------------------------------------------------
    # Internals
    # ---------------------------------------------------------
    def _on_collision(self, event):
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.collision_hist.append(intensity)

    def _process_camera(self, image):
        """
        Convert CARLA semseg raw_data to (3,H,W) in self.front_image.
        """
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        seg = array[:, :, 2]  # channel 2 is the class ID in semantic seg
        seg_3c = np.stack([seg, seg, seg], axis=-1)  # shape (H,W,3)
        seg_3c = seg_3c.astype(np.uint8)
        # transpose to (3,H,W)
        seg_3c = np.transpose(seg_3c, (2,0,1))
        self.front_image = seg_3c

        # also update the camera transform in case it moves with the vehicle
        if self.camera_sensor:
            self.camera_transform = self.camera_sensor.get_transform()

    def _compute_reward_done(self):
        done = False
        # Collision => -30, done
        if len(self.collision_hist) > 0:
            return -30.0, True

        if self.episode_step >= self.max_episode_steps:
            done = True

        # Quick shaping
        vec_obs = self._get_vector_obs()
        _, _, _, _, speed, lane_off, heading_err = vec_obs
        target_speed = 10.0
        heading_pen = -0.1 * abs(heading_err)
        offset_pen = -0.05 * abs(lane_off)
        speed_pen = -0.01 * abs(speed - target_speed)
        step_reward = heading_pen + offset_pen + speed_pen
        # clamp to [-1,1]
        step_reward = float(np.clip(step_reward, -1.0, 1.0))

        return step_reward, done

    def _get_obs(self):
        """
        Return Dict with:
          "image": self.front_image (3,H,W)
          "vector": 7D array => [next_wp_x, next_wp_y, curr_x, curr_y, speed, lane_offset, heading_err]
        """
        return {
            "image": self.front_image,
            "vector": self._get_vector_obs()
        }

    def _get_vector_obs(self):
        curr_tf = self.vehicle.get_transform()
        vx = curr_tf.location.x
        vy = curr_tf.location.y

        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        lane_off, heading_err = self._lane_center_offset_and_heading(curr_tf)

        # Next WP ~5m ahead
        waypoint = self.map.get_waypoint(curr_tf.location, project_to_road=True)
        if waypoint is not None:
            next_wps = waypoint.next(5.0)
            if len(next_wps) > 0:
                nx = next_wps[0].transform.location.x
                ny = next_wps[0].transform.location.y
            else:
                nx, ny = vx, vy
        else:
            nx, ny = vx, vy

        return np.array([nx, ny, vx, vy, speed, lane_off, heading_err], dtype=np.float32)

    def _lane_center_offset_and_heading(self, vehicle_transform):
        waypoint = self.map.get_waypoint(
            vehicle_transform.location, project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if waypoint is None:
            return 0.0, 0.0

        lane_tf = waypoint.transform
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
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        if self.screen:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None


def main():
    # Environment config
    config = {
        "host": "localhost",
        "port": 2000,
        "timeout": 10.0,
        "synchronous_mode": True,
        "img_width": 84,
        "img_height": 84,
        "use_render": True  # We'll enable PyGame render
    }

    # Create env
    def make_env():
        return CarlaMultiInputEnv(config)

    env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    # Optional callback for evaluation
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs_multiinput_lstm/",
        log_path="./logs_multiinput_lstm/",
        eval_freq=50_000,
        deterministic=True,
        render=False
    )

    # Large network via policy_kwargs => big CNN, big LSTM
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
            cnn_extractor_kwargs=dict(
                features_dim=512,
            ),
        )
    )

    # Train
    model.learn(
        total_timesteps=200000,  # Adjust as needed
        callback=eval_callback
    )
    model.save("multiinput_cnn_lstm_carla_pygame")
    print("Training complete!")


if __name__ == "__main__":
    main()
