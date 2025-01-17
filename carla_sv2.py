#!/usr/bin/env python

import gym
import numpy as np
import cv2
import random
import math
import carla

from gym import spaces

# For Recurrent PPO and the LSTM-based policy
from sb3_contrib import RecurrentPPO
# This policy can handle Dict observation spaces with an LSTM
from sb3_contrib.ppo_recurrent.policies import MultiInputLstmPolicy

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback


class CarlaMultiInputEnv(gym.Env):
    """
    A CARLA environment with:
      - Semantic segmentation camera -> shape=(3, H, W) for the 'image' subspace
      - Additional vector obs -> shape=(7,) for the 'vector' subspace
      - Action space: [steer, throttle] with bounds in [-0.5, 0.5] and [0.0, 0.5], respectively
      - Episode length: 2500 steps
      - Reward in [-1, 1] each step, -30 if collision
    """

    def __init__(self, config):
        super(CarlaMultiInputEnv, self).__init__()

        self.host = config.get("host", "localhost")
        self.port = config.get("port", 2000)
        self.timeout = config.get("timeout", 10.0)
        self.synchronous_mode = config.get("synchronous_mode", True)

        self.img_width = config.get("img_width", 84)
        self.img_height = config.get("img_height", 84)

        self.max_episode_steps = 2500  # Per user requirement
        self.episode_step = 0
        self.vehicle = None
        self.collision_sensor = None
        self.camera_sensor = None

        # Connect to CARLA
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()
        self.map = self.world.get_map()

        if self.synchronous_mode:
            settings = self.world.get_settings()
            settings.fixed_delta_seconds = 0.05  # 20 FPS
            settings.synchronous_mode = True
            self.world.apply_settings(settings)

        self.bp_library = self.world.get_blueprint_library()

        # Action space: 2D continuous [steer, throttle]
        # steer in [-0.5, 0.5], throttle in [0.0, 0.5]
        self.action_space = spaces.Box(
            low=np.array([-0.5, 0.0], dtype=np.float32),
            high=np.array([0.5, 0.5], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        # Observation space: Dict with two keys: "image" and "vector"
        # "image": shape=(3, H, W) channel-first for SB3 CNN
        # "vector": shape=(7,) => [next_wp_x, next_wp_y, curr_x, curr_y, speed, lane_offset, heading_err]
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

        # For storing camera images & collisions
        self.collision_hist = []
        self.front_image = np.zeros((3, self.img_height, self.img_width), dtype=np.uint8)

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

        # Semantic Segmentation camera
        cam_bp = self.bp_library.find('sensor.camera.semantic_segmentation')
        cam_bp.set_attribute("image_size_x", f"{self.img_width}")
        cam_bp.set_attribute("image_size_y", f"{self.img_height}")
        cam_bp.set_attribute("fov", "90")
        cam_transform = carla.Transform(carla.Location(x=1.6, z=2.0))
        self.camera_sensor = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.vehicle)
        self.camera_sensor.listen(lambda img: self._process_camera(img))

        # Let things stabilize
        for _ in range(20):
            if self.synchronous_mode:
                self.world.tick()
            else:
                self.world.wait_for_tick()

        return self._get_obs()

    def step(self, action):
        self.episode_step += 1

        # Action => VehicleControl
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
        reward, done = self._compute_reward_and_done()

        return obs, reward, done, {}

    def _get_obs(self):
        """
        Returns a Dict:
          'image': (3, H, W) semantic seg image
          'vector': 7 floats (next_wp_x, next_wp_y, curr_x, curr_y, speed, lane_offset, heading_err)
        """
        # 1) Image
        img_obs = self.front_image  # shape=(3, H, W), already processed

        # 2) Vector
        curr_transform = self.vehicle.get_transform()
        curr_loc = curr_transform.location
        curr_x, curr_y = curr_loc.x, curr_loc.y

        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        lane_offset, heading_err = self._lane_center_offset_and_heading(curr_transform)

        # Next waypoint 5m ahead
        waypoint = self.map.get_waypoint(curr_loc, project_to_road=True)
        if waypoint is not None:
            next_wps = waypoint.next(5.0)
            if len(next_wps) > 0:
                nx_loc = next_wps[0].transform.location
                next_wp_x, next_wp_y = nx_loc.x, nx_loc.y
            else:
                # fallback
                next_wp_x, next_wp_y = curr_x, curr_y
        else:
            next_wp_x, next_wp_y = curr_x, curr_y

        vec_obs = np.array([
            next_wp_x, next_wp_y,
            curr_x, curr_y,
            speed,
            lane_offset,
            heading_err
        ], dtype=np.float32)

        return {
            "image": img_obs,
            "vector": vec_obs
        }

    def _compute_reward_and_done(self):
        done = False

        # Collision => -30 and done
        if len(self.collision_hist) > 0:
            return -30.0, True

        if self.episode_step >= self.max_episode_steps:
            done = True

        # Basic shaping: keep lane_offset, heading_err small, speed near ~10 m/s
        transform = self.vehicle.get_transform()
        lane_offset, heading_err = self._lane_center_offset_and_heading(transform)
        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        target_speed = 10.0  # m/s

        # Some small penalty terms
        heading_pen = -0.1 * abs(heading_err)
        offset_pen = -0.05 * abs(lane_offset)
        speed_pen = -0.01 * abs(speed - target_speed)

        step_reward = heading_pen + offset_pen + speed_pen

        # clamp to [-1, 1]
        step_reward = float(np.clip(step_reward, -1.0, 1.0))
        return step_reward, done

    def _lane_center_offset_and_heading(self, vehicle_transform):
        """
        Return (lane_offset, heading_error in [-pi, pi])
        """
        waypoint = self.map.get_waypoint(
            vehicle_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if waypoint is None:
            return 0.0, 0.0

        lane_transform = waypoint.transform
        vx, vy = vehicle_transform.location.x, vehicle_transform.location.y
        vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)

        lx, ly = lane_transform.location.x, lane_transform.location.y
        lane_yaw = math.radians(lane_transform.rotation.yaw)

        right_vec = lane_transform.get_right_vector()
        dx, dy = vx - lx, vy - ly
        lane_offset = dx * right_vec.x + dy * right_vec.y

        heading_err = vehicle_yaw - lane_yaw
        heading_err = (heading_err + math.pi) % (2 * math.pi) - math.pi
        return lane_offset, heading_err

    def _on_collision(self, event):
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.collision_hist.append(intensity)

    def _process_camera(self, image):
        """
        Semantic segmentation camera => each pixel’s G-channel is class index
        We'll map it to (3, H, W) for CNN, replicating the index in 3 channels or color mapping.
        """
        # Convert raw data to (H, W, 4)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))

        # For semseg in CARLA, channel 2 (BGR) often holds the semantic class ID
        seg = array[:, :, 2]  # shape (H, W), 0..22 range
        # Expand to 3 channels
        seg_3c = np.stack([seg, seg, seg], axis=-1)  # shape (H, W, 3)

        # Transpose to (3, H, W) for SB3
        seg_3c = np.transpose(seg_3c, (2, 0, 1))  # shape=(3, H, W)
        self.front_image = seg_3c.astype(np.uint8)

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


def main():
    config = {
        "host": "localhost",
        "port": 2000,
        "timeout": 10.0,
        "synchronous_mode": True,
        "img_width": 84,
        "img_height": 84
    }

    # Create the environment
    def make_env():
        return CarlaMultiInputEnv(config)

    env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    # Optional: Evaluate performance periodically
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs_multiinput_lstm/",
        log_path="./logs_multiinput_lstm/",
        eval_freq=50_000,
        deterministic=True,
        render=False
    )

    # Create RecurrentPPO model using MultiInputLstmPolicy with a LARGE network
    # The big network is controlled by:
    #   - net_arch (size of MLP part in pi/vf)
    #   - n_lstm_layers, lstm_hidden_size
    #   - cnn_extractor_kwargs (bigger CNN, more filters, etc.)
    model = RecurrentPPO(
        policy=MultiInputLstmPolicy,
        env=env,
        n_steps=128,       # Rollout length
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
        # Here we define the large architecture
        policy_kwargs=dict(
            # The LSTM settings
            n_lstm_layers=2,          # more than 1 layer
            lstm_hidden_size=512,     # bigger hidden layer
            # net_arch => how the extracted features are processed inside the policy
            # We pass separate arches for pi and vf
            net_arch=[dict(pi=[512, 512, 512],
                           vf=[512, 512, 512])],
            # If you want a bigger CNN, you can define:
            cnn_extractor_kwargs=dict(
                features_dim=512,     # final CNN embedding dimension
                # You can also pass "conv_arch=[(32,8,4), (64,4,2), ...]" in some SB3 versions
            ),
        )
    )

    model.learn(
        total_timesteps=500000,
        callback=eval_callback
    )

    model.save("multiinput_cnn_lstm_carla")
    print("Training complete! Model saved to multiinput_cnn_lstm_carla.zip")

if __name__ == "__main__":
    main()
