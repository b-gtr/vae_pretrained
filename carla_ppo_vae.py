"""
File: carla_ppo_vae.py

Example for autonomous driving in CARLA using:
 - A pretrained VAE for feature extraction
 - Stable-Baselines3 PPO for policy optimization
 - A custom Gym environment that wraps CARLA’s Python API

Adjust to your environment setup (map, spawn, sensors, etc.).
"""

import os
import time
import math
import numpy as np
from collections import deque

import gym
import torch
import torch.nn as nn

import carla
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# -------------------------------
# 1) Import your VAE code
# -------------------------------
# We assume you have vae.py, encoder.py, decoder.py with a class "VariationalAutoencoder"
# that loads your 95-dim latent model. Adjust import if different.
from vae import VariationalAutoencoder

# If you keep each part separate, you could do:
# from encoder import VariationalEncoder
# from decoder import Decoder
# But typically you only need the combined class.


# -------------------------------
# 2) Carla Environment
# -------------------------------
class CarlaEnv(gym.Env):
    """
    A Gym-like environment for CARLA, with:
     - A semantic-segmentation camera (160x80)
     - A collision sensor
     - Observations:
       * VAE latent vector (z_dim=95)
       * External features: speed, last_steer, dist_from_waypoint, orientation_diff, throttle
     - Action: [steer, throttle], both continuous
     - Reward: see _compute_reward() per your described formula
     - Termination: off-lane, collisions, too slow, success, etc.
    """

    def __init__(self,
                 vae_model_path="autoencoder/model/var_autoencoder.pth",
                 z_dim=95,
                 host="127.0.0.1",
                 port=2000,
                 town="Town07",
                 max_episode_steps=7500,
                 target_speed=20.0,
                 max_distance=3.0,   # 3m from lane center
                 max_angle_deg=20.0  # orientation threshold
                 ):
        super(CarlaEnv, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load your VAE (Encoder+Decoder)
        self.z_dim = z_dim
        self.vae = VariationalAutoencoder(latent_dims=z_dim).to(self.device)
        self.vae.load_state_dict(torch.load(vae_model_path, map_location=self.device))
        self.vae.eval()
        # Freeze VAE parameters
        for param in self.vae.parameters():
            param.requires_grad = False

        # Connect to CARLA
        self.client = carla.Client(host, port)
        self.client.set_timeout(5.0)
        self.world = self.client.load_world(town)

        # Asynchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = False  # asynchronous
        self.world.apply_settings(settings)

        self.max_episode_steps = max_episode_steps
        self.episode_steps = 0

        # For reward shaping
        self.target_speed = target_speed
        self.max_distance = max_distance
        self.max_angle_deg = max_angle_deg

        # GYM: observation + action space
        # Observations: z_dim + 5 external features
        obs_dim = self.z_dim + 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Action: [steer, throttle]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            shape=(2,),
            dtype=np.float32
        )

        # Actor/pointer to car + sensors
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None

        # Queues for sensor data
        self.image_queue = deque(maxlen=1)
        self.collision_queue = deque(maxlen=1)

        # We track the last steer for external features
        self.last_steer = 0.0

        # Initialize the scene
        self._setup_carla()

    def _setup_carla(self):
        """Spawn the vehicle, camera, collision sensor, etc."""
        self._clean_world()

        blueprint_library = self.world.get_blueprint_library()

        # Spawn vehicle
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = np.random.choice(spawn_points)  # random or define your own
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)

        # Camera: semantic_segmentation
        cam_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        cam_bp.set_attribute("image_size_x", "160")
        cam_bp.set_attribute("image_size_y", "80")
        cam_bp.set_attribute("fov", "110")
        cam_transform = carla.Transform(carla.Location(x=2.0, z=1.4))
        self.camera = self.world.try_spawn_actor(cam_bp, cam_transform, attach_to=self.vehicle)

        # Collision sensor
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.try_spawn_actor(collision_bp, cam_transform, attach_to=self.vehicle)

        # Register callbacks
        self.camera.listen(lambda data: self._on_cam_data(data))
        self.collision_sensor.listen(lambda event: self._on_collision(event))

    def _clean_world(self):
        """Destroy leftover actors from old runs."""
        actors = self.world.get_actors()
        for actor in actors:
            if 'vehicle.' in actor.type_id or 'sensor.' in actor.type_id:
                actor.destroy()

    def _on_cam_data(self, image):
        """
        Callback for the semantic-segmentation camera.
        By default, image.raw_data is BGRA format: shape (H x W x 4).
        Typically, the 'G' or 'R' channel encodes the class ID in CARLA's semantic camera.
        We'll just do [0..1] normalized.
        """
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        # E.g., take the "G" channel (index=2). Some setups might differ.
        seg_img = array[:, :, 2].astype(np.float32) / 255.0
        self.image_queue.append(seg_img)

    def _on_collision(self, event):
        self.collision_queue.append(event)

    def reset(self):
        """Reset the environment."""
        self.episode_steps = 0
        self.last_steer = 0.0
        self._clean_world()
        self._setup_carla()

        # Wait a bit so that sensors produce data
        for _ in range(10):
            # if synchronous: self.world.tick()
            time.sleep(0.05)

        return self._get_observation()

    def step(self, action):
        self.episode_steps += 1

        steer, throttle = action
        steer = float(np.clip(steer, -1.0, 1.0))
        throttle = float(np.clip(throttle, 0.0, 1.0))

        # Apply control
        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = 0.0
        self.vehicle.apply_control(control)

        self.last_steer = steer

        # if synchronous: self.world.tick()
        obs = self._get_observation()
        reward = self._compute_reward()
        done = self._check_done()
        info = {}

        return obs, reward, done, info

    def _get_observation(self):
        """
        Build observation from:
         - semantic seg image -> VAE -> latent z
         - external features: speed, last_steer, dist_from_lane, angle_diff, throttle
        """
        if len(self.image_queue) == 0:
            # if no image yet, fallback
            seg_img = np.zeros((80, 160), dtype=np.float32)
        else:
            seg_img = self.image_queue[-1]

        # Convert to Torch
        img_tensor = torch.from_numpy(seg_img).unsqueeze(0).unsqueeze(0).float().to(self.device)
        # shape: (1,1,80,160)

        with torch.no_grad():
            # This calls: encoder -> sample z
            z = self.vae.encoder(img_tensor)  # shape (1, z_dim)
        z_np = z.squeeze(0).cpu().numpy()  # shape (z_dim,)

        # External features
        speed = self._get_speed()
        throttle = self.vehicle.get_control().throttle
        dist_lane = self._dist_from_lane_center()
        angle_diff = self._orientation_diff()

        # Scale speed to [0..1] based on target_speed=20 km/h
        speed_norm = min(speed / self.target_speed, 1.0)
        dist_norm = dist_lane / self.max_distance
        angle_norm = angle_diff / self.max_angle_deg

        ext_feats = np.array([
            speed_norm,
            self.last_steer,
            dist_norm,
            angle_norm,
            throttle
        ], dtype=np.float32)

        obs = np.concatenate([z_np, ext_feats], axis=0)  # shape = (z_dim + 5,)
        return obs

    def _get_speed(self):
        """Get vehicle speed in km/h."""
        vel = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        return speed

    def _dist_from_lane_center(self):
        """Lateral distance from waypoint center (in meters)."""
        transform = self.vehicle.get_transform()
        location = transform.location
        waypoint = self.world.get_map().get_waypoint(location, project_to_road=True)
        if waypoint is None:
            return self.max_distance * 2  # big => out of road
        lane_center = waypoint.transform.location
        dx = location.x - lane_center.x
        dy = location.y - lane_center.y
        return math.sqrt(dx*dx + dy*dy)

    def _orientation_diff(self):
        """Angle difference between vehicle heading and lane heading (0..180)."""
        vehicle_yaw = self.vehicle.get_transform().rotation.yaw
        waypoint = self.world.get_map().get_waypoint(self.vehicle.get_transform().location)
        if waypoint is None:
            return 180.0
        lane_yaw = waypoint.transform.rotation.yaw
        diff = abs(vehicle_yaw - lane_yaw) % 360
        if diff > 180:
            diff = 360 - diff
        return diff

    def _compute_reward(self):
        """
        A simplified version of Equation 3.2 from your text:
         - -10 on collision / out-of-lane
         - scale speed, distance, angle for reward
        """
        # Collision
        if len(self.collision_queue) > 0:
            return -10.0

        # Distance > 3m => out
        dist_lane = self._dist_from_lane_center()
        if dist_lane > self.max_distance:
            return -10.0

        speed = self._get_speed()
        angle_diff = self._orientation_diff()

        # angle reward
        if angle_diff < self.max_angle_deg:
            a_rew = 1.0 - (angle_diff / self.max_angle_deg)
        else:
            a_rew = 0.0

        # dist factor in [0..1]
        d_norm = dist_lane / self.max_distance
        d_term = (1.0 - d_norm)

        # Speed term
        v_min = 5.0
        v_target = self.target_speed
        v_max = 30.0

        if speed < v_min:
            speed_term = speed / v_min
        elif speed < v_target:
            speed_term = 1.0
        else:
            # if speed > target
            speed_term = 1.0 - (speed - v_target)/(v_max - v_target)
            speed_term = max(speed_term, 0.0)

        reward = speed_term * d_term * a_rew
        return float(reward)

    def _check_done(self):
        """Episode termination logic."""
        # max steps
        if self.episode_steps >= self.max_episode_steps:
            return True

        # collision
        if len(self.collision_queue) > 0:
            return True

        # out of lane
        if self._dist_from_lane_center() > self.max_distance:
            return True

        # if speed < 1.0 after some time
        if self._get_speed() < 1.0 and self.episode_steps > 100:
            return True

        # You can add success check if you want (reaching a destination, etc.)
        return False

    def close(self):
        self._clean_world()


# -------------------------------
# 3) Training with Stable-Baselines3 PPO
# -------------------------------
def main():
    # a) create the environment
    env = CarlaEnv(
        vae_model_path="autoencoder/model/var_autoencoder.pth",  # adjust path if needed
        z_dim=95,
        host="127.0.0.1",
        port=2000,
        town="Town07",
        max_episode_steps=7500,
        target_speed=20.0,
        max_distance=3.0,
        max_angle_deg=20.0
    )

    # b) define policy network architecture = [500,300,100]
    #   using Tanh as described
    policy_kwargs = dict(
        net_arch=[dict(pi=[500, 300, 100], vf=[500, 300, 100])],
        activation_fn=nn.Tanh
    )

    # c) create PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=1024,         # how many steps to run per update
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.0,         # optionally tune
        vf_coef=0.5,          # scaling for value loss
        policy_kwargs=policy_kwargs,
        verbose=1
    )

    # d) optional: checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=20000, 
        save_path="./logs/",
        name_prefix="carla_ppo_ckpt"
    )

    # e) train
    total_timesteps = 1_000_000  # adjust
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # f) save final model
    model.save("ppo_carla_semseg_vae_final.zip")
    print("Training finished and model saved.")

    # g) test the policy
    obs = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
    print("Test episode reward:", total_reward)

    env.close()


if __name__ == "__main__":
    main()
