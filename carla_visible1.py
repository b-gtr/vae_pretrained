#!/usr/bin/env python

import gym
import numpy as np
import random
import math
import carla
import pygame
import sys

from gym import spaces

# Recurrent PPO with LSTM policy that supports multi-input
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MultiInputLstmPolicy

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

####################################################################################
# 1) A simple color palette for semantic segmentation classes in CARLA
####################################################################################
SEMSEG_PALETTE = {
    0:  (  0,   0,   0),     # Unlabeled
    1:  ( 70,  70,  70),     # Building
    2:  (190, 153, 153),     # Fence
    3:  ( 72,   0,  90),     # Other
    4:  (220,  20,  60),     # Pedestrian
    5:  (153, 153, 153),     # Pole
    6:  (157, 234,  50),     # RoadLines
    7:  (128,  64, 128),     # Road
    8:  (244,  35, 232),     # Sidewalk
    9:  (107, 142,  35),     # Vegetation
    10: (  0,   0, 142),     # Vehicle
    11: (102, 102, 156),     # Wall
    12: (220, 220,   0),     # TrafficSign
    # Add more if you need them...
}

def apply_semseg_palette(semseg_id_img: np.ndarray) -> np.ndarray:
    """
    Map each pixel's class ID to a color using SEMSEG_PALETTE.
    Output shape: (H, W, 3).
    """
    height, width = semseg_id_img.shape[:2]
    colored = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id, color in SEMSEG_PALETTE.items():
        colored[semseg_id_img == class_id] = color
    return colored

####################################################################################
# 2) Utility for projecting a world location to camera pixel coords
####################################################################################
def world_to_camera(world_loc, camera_transform, camera_intrinsic):
    """
    Approximate transform from 3D world location to 2D camera coords.
    Returns (u, v) or None if behind camera.
    """
    import numpy as np

    # Build inverse of (camera->world)
    c_yaw = math.radians(camera_transform.rotation.yaw)
    c_pitch = math.radians(camera_transform.rotation.pitch)
    c_roll = math.radians(camera_transform.rotation.roll)

    R_yaw = np.array([[ math.cos(c_yaw), -math.sin(c_yaw), 0],
                      [ math.sin(c_yaw),  math.cos(c_yaw), 0],
                      [             0,               0,   1]])
    R_pitch = np.array([[ math.cos(c_pitch), 0, math.sin(c_pitch)],
                        [                0,   1,               0],
                        [-math.sin(c_pitch), 0, math.cos(c_pitch)]])
    R_roll = np.array([[1,              0,               0],
                       [0, math.cos(c_roll), -math.sin(c_roll)],
                       [0, math.sin(c_roll),  math.cos(c_roll)]])
    R = R_roll @ R_pitch @ R_yaw

    t = np.array([camera_transform.location.x,
                  camera_transform.location.y,
                  camera_transform.location.z])

    R_inv = R.T
    t_inv = -R_inv @ t
    transform_mat = np.eye(4)
    transform_mat[0:3, 0:3] = R_inv
    transform_mat[0:3, 3]   = t_inv

    # Convert the world_loc (carla.Location) to a homogeneous vector
    world_point = np.array([world_loc.x, world_loc.y, world_loc.z, 1])
    cam_coords = transform_mat @ world_point

    if cam_coords[2] <= 0.01:
        return None

    # Project
    px = (camera_intrinsic[0, 0] * (cam_coords[0] / cam_coords[2])) + camera_intrinsic[0, 2]
    py = (camera_intrinsic[1, 1] * (cam_coords[1] / cam_coords[2])) + camera_intrinsic[1, 2]
    return (int(px), int(py))

####################################################################################
# 3) The CARLA Environment with Multi-Input observation and PyGame Render
####################################################################################
class CarlaMultiInputEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 2000)
        self.timeout = config.get("timeout", 10.0)
        self.synchronous_mode = config.get("synchronous_mode", True)

        self.img_width  = config.get("img_width",  84)
        self.img_height = config.get("img_height", 84)
        self.max_episode_steps = 2500
        self.episode_step = 0

        # Connect to CARLA
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()
        self.map   = self.world.get_map()

        if self.synchronous_mode:
            settings = self.world.get_settings()
            settings.fixed_delta_seconds = 0.05
            settings.synchronous_mode = True
            self.world.apply_settings(settings)

        self.bp_lib = self.world.get_blueprint_library()

        # Action space: [steer, throttle]
        self.action_space = spaces.Box(
            low=np.array([-0.5, 0.0], dtype=np.float32),
            high=np.array([0.5,  0.5], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        # Observation space: Dict{"image":(3,H,W), "vector":(7,)}
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

        # Actors & sensors
        self.vehicle = None
        self.collision_sensor = None
        self.camera_sensor = None
        self.collision_hist = []
        self.front_image = np.zeros((3, self.img_height, self.img_width), dtype=np.uint8)
        self.camera_transform = None

        # PyGame
        self.use_render = config.get("use_render", True)
        self.screen = None
        self.clock  = None
        self.font   = None

        # Build approximate pinhole intrinsic
        self.intrinsic = self._build_intrinsic()

    def _build_intrinsic(self):
        import numpy as np
        f_x = self.img_width / (2.0 * math.tan(math.radians(90.0)/2.0))
        f_y = f_x
        c_x = self.img_width / 2.0
        c_y = self.img_height / 2.0
        K = np.array([
            [f_x,   0,    c_x],
            [ 0,    f_y,  c_y],
            [ 0,    0,     1 ]
        ])
        return K

    def reset(self):
        self._destroy_actors()
        self.collision_hist.clear()
        self.episode_step = 0

        # Spawn vehicle
        spawn_points = self.map.get_spawn_points()
        if not spawn_points:
            raise ValueError("No spawn points in map!")
        spawn_pt = random.choice(spawn_points)
        bp = self.bp_lib.find('vehicle.tesla.model3')
        self.vehicle = None
        for _ in range(10):
            actor = self.world.try_spawn_actor(bp, spawn_pt)
            if actor:
                self.vehicle = actor
                break
        if not self.vehicle:
            raise ValueError("Couldn't spawn vehicle after multiple attempts.")

        # Collision sensor
        col_bp = self.bp_lib.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda e: self._on_collision(e))

        # Semantic seg camera
        cam_bp = self.bp_lib.find('sensor.camera.semantic_segmentation')
        cam_bp.set_attribute("image_size_x", f"{self.img_width}")
        cam_bp.set_attribute("image_size_y", f"{self.img_height}")
        cam_bp.set_attribute("fov", "90")
        cam_tf = carla.Transform(carla.Location(x=1.6, z=2.0))
        self.camera_sensor = self.world.spawn_actor(cam_bp, cam_tf, attach_to=self.vehicle)
        self.camera_sensor.listen(lambda img: self._process_camera(img))

        self.camera_transform = self.camera_sensor.get_transform()

        # Let simulation stabilize
        for _ in range(20):
            if self.synchronous_mode:
                self.world.tick()
            else:
                self.world.wait_for_tick()

        if self.use_render and self.screen is None:
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((800,600))
            self.clock = pygame.time.Clock()
            self.font  = pygame.font.SysFont("Arial", 20)

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
        Must be called repeatedly to keep the PyGame window responsive.
        """
        if not self.use_render or self.screen is None:
            return

        # Process PyGame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Convert self.front_image (3,H,W)->(H,W,3)
        img = np.transpose(self.front_image, (1,2,0))
        surf = pygame.surfarray.make_surface(img)
        # Scale to bigger
        camera_surf = pygame.transform.scale(surf, (400,400))

        self.screen.fill((0,0,0))
        self.screen.blit(camera_surf, (0,0))

        # Show text info
        vec_obs = self._get_vector_obs()
        nx, ny, cx, cy, speed, lane_off, heading_err = vec_obs
        lines = [
            f"Step: {self.episode_step}/{self.max_episode_steps}",
            f"Speed: {speed:.2f} m/s",
            f"Lane Off: {lane_off:.2f}",
            f"Heading Err: {heading_err:.2f}",
            f"Pos: ({cx:.1f}, {cy:.1f})",
            f"NextWP: ({nx:.1f}, {ny:.1f})"
        ]
        x_text, y_text = 420, 10
        for line in lines:
            text_surf = self.font.render(line, True, (255,255,255))
            self.screen.blit(text_surf, (x_text,y_text))
            y_text += 25

        # Draw a dot for the next waypoint if in front
        from carla import Location
        waypoint_loc = Location(x=nx, y=ny, z=0.0)
        dot = world_to_camera(waypoint_loc, self.camera_transform, self.intrinsic)
        if dot:
            px, py = dot
            # scale to the 400x400 display
            scale_x = 400 / self.img_width
            scale_y = 400 / self.img_height
            dx, dy = int(px*scale_x), int(py*scale_y)
            if 0 <= dx < 400 and 0 <= dy < 400:
                pygame.draw.circle(camera_surf, (255,0,0), (dx,dy), 5)
        self.screen.blit(camera_surf, (0,0))

        pygame.display.flip()
        self.clock.tick(30)

    # --------------------- Internals ---------------------
    def _on_collision(self, event):
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.collision_hist.append(intensity)

    def _process_camera(self, image):
        import numpy as np
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        sem_id = array[:,:,2]  # semantic class ID
        colored = apply_semseg_palette(sem_id)
        # shape (H,W,3), then transpose to (3,H,W)
        colored_t = np.transpose(colored, (2,0,1))
        self.front_image = colored_t
        if self.camera_sensor:
            self.camera_transform = self.camera_sensor.get_transform()

    def _compute_reward_done(self):
        done = False
        if len(self.collision_hist) > 0:
            return -30.0, True
        if self.episode_step >= self.max_episode_steps:
            done = True

        # Basic shaping: speed near 10 m/s, minimal lane offset/heading error
        vec = self._get_vector_obs()
        _, _, _, _, speed, lane_off, heading_err = vec
        target_speed = 10.0
        heading_pen  = -0.1*abs(heading_err)
        offset_pen   = -0.05*abs(lane_off)
        speed_pen    = -0.01*abs(speed - target_speed)
        step_reward  = heading_pen + offset_pen + speed_pen
        step_reward  = float(np.clip(step_reward, -1, 1))
        return step_reward, done

    def _get_obs(self):
        return {
            "image": self.front_image,
            "vector": self._get_vector_obs()
        }

    def _get_vector_obs(self):
        tf = self.vehicle.get_transform()
        vx, vy = tf.location.x, tf.location.y
        vel = self.vehicle.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        off, hed = self._lane_center_offset_and_heading(tf)

        # Next WP 5m ahead
        wp = self.map.get_waypoint(tf.location, project_to_road=True)
        if wp:
            nxts = wp.next(5.0)
            if len(nxts)>0:
                nx, ny = nxts[0].transform.location.x, nxts[0].transform.location.y
            else:
                nx, ny = vx, vy
        else:
            nx, ny = vx, vy

        return np.array([nx, ny, vx, vy, speed, off, hed], dtype=np.float32)

    def _lane_center_offset_and_heading(self, vehicle_transform):
        wp = self.map.get_waypoint(vehicle_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if not wp:
            return 0.0, 0.0
        lane_tf = wp.transform
        vx, vy = vehicle_transform.location.x, vehicle_transform.location.y
        vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)
        lx, ly = lane_tf.location.x, lane_tf.location.y
        lane_yaw = math.radians(lane_tf.rotation.yaw)

        rvec = lane_tf.get_right_vector()
        dx, dy = vx-lx, vy-ly
        lane_off  = dx*rvec.x + dy*rvec.y

        hed_err = vehicle_yaw - lane_yaw
        hed_err = (hed_err + math.pi) % (2*math.pi) - math.pi
        return lane_off, hed_err

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
            self.font  = None


####################################################################################
# 4) Main: Train and Test with Render
####################################################################################
def main():
    config = {
        "host": "localhost",
        "port": 2000,
        "timeout": 10.0,
        "synchronous_mode": True,
        "img_width":  84,
        "img_height": 84,
        "use_render": False  # Turn off rendering during training to avoid slowdowns
    }

    # Vectorized env for training
    def make_env():
        return CarlaMultiInputEnv(config)
    env = DummyVecEnv([make_env])

    # Another env for evaluation
    eval_env = DummyVecEnv([make_env])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs_multiinput_lstm/",
        log_path="./logs_multiinput_lstm/",
        eval_freq=50000,
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

    # -------------- Train (no rendering) ---------------
    model.learn(total_timesteps=200000, callback=eval_callback)
    model.save("multiinput_cnn_lstm_carla_pygame")
    print("Training complete!")

    # -------------- Test with rendering ---------------
    # We'll create a new env that has use_render=True
    config["use_render"] = True
    test_env = CarlaMultiInputEnv(config)

    obs = test_env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    for _ in range(1000):
        # Let the model decide
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        
        # Step
        obs, reward, done, info = test_env.step(action[0])
        
        # Render
        test_env.render()  # <= IMPORTANT, must be called repeatedly
        
        episode_starts = done
        if done:
            print("Episode done! Reward:", reward)
            obs = test_env.reset()
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)

    test_env.close()


if __name__ == "__main__":
    main()
