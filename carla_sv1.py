#!/usr/bin/env python

import gym
import numpy as np
import carla
import random
import time
import math
import cv2

from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ----------------------------------------------------------------------------------
# 1) Custom CARLA Gym Environment
# ----------------------------------------------------------------------------------
class CarlaEnv(gym.Env):
    """
    A custom Gym environment for autonomous driving in CARLA.
    Observations:
        - For simplicity: a front camera image (e.g., 84x84x3) or
          a vector containing speed, distance to lane center, etc.
    Actions:
        - [steer, throttle, brake]
    Reward:
        - Based on staying in lane, avoiding collisions, comfortable driving, etc.
    """
    def __init__(self, config):
        super(CarlaEnv, self).__init__()
        
        # Carla server configuration
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 2000)
        self.timeout = config.get("timeout", 10.0)
        
        # Connect to CARLA
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()
        
        # Optionally set synchronous mode
        self.synchronous_mode = config.get("synchronous_mode", False)
        if self.synchronous_mode:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05  # 20 FPS
            self.world.apply_settings(settings)
        
        # Spawn ego vehicle
        self.blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = self.blueprint_library.find('vehicle.tesla.model3')
        
        # Spawn point
        spawn_points = self.world.get_map().get_spawn_points()
        self.spawn_point = random.choice(spawn_points)
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, self.spawn_point)
        
        # Collision sensor
        col_sensor_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            col_sensor_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.collision_hist = []
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        
        # Camera sensor (if needed for image-based RL)
        self.img_height = 84
        self.img_width = 84
        self.camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', f'{self.img_width}')
        self.camera_bp.set_attribute('image_size_y', f'{self.img_height}')
        self.camera_bp.set_attribute('fov', '90')
        self.camera_init_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera_sensor = self.world.spawn_actor(
            self.camera_bp,
            self.camera_init_transform,
            attach_to=self.vehicle
        )
        self.front_camera = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        self.camera_sensor.listen(lambda image: self._process_camera(image))
        
        # Observation & Action space
        # Here, we define an example for image-based + speed-based observation
        # If you just want a vector observation, adapt accordingly.
        # Observations: stack of camera image + speed
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.img_height, self.img_width, 3), dtype=np.uint8
        )
        
        # Actions: [steer, throttle, brake]
        # steer in [-1, 1], throttle in [0, 1], brake in [0, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            shape=(3,),
            dtype=np.float32
        )
        
        # Additional fields
        self.max_episode_steps = config.get("max_episode_steps", 1000)
        self.episode_step = 0
        self.max_speed = 30.0  # m/s
        self.speed_weight = config.get("speed_weight", 0.1)
        
    def reset(self):
        """Resets the environment for a new episode."""
        self.episode_step = 0
        
        # Clean up the old vehicle and sensors
        self._destroy_actors()
        
        # Respawn ego vehicle
        self.vehicle = self.world.try_spawn_actor(
            self.blueprint_library.find('vehicle.tesla.model3'),
            random.choice(self.world.get_map().get_spawn_points())
        )
        
        # Respawn collision sensor
        col_sensor_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            col_sensor_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.collision_hist = []
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        
        # Respawn camera
        self.camera_sensor = self.world.spawn_actor(
            self.camera_bp,
            self.camera_init_transform,
            attach_to=self.vehicle
        )
        self.front_camera = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        self.camera_sensor.listen(lambda image: self._process_camera(image))
        
        # Wait a bit to stabilize
        for _ in range(10):
            if self.synchronous_mode:
                self.world.tick()
            else:
                self.world.wait_for_tick()
        
        # Return initial observation
        return self._get_observation()
    
    def step(self, action):
        """Executes one step in the environment."""
        self.episode_step += 1
        
        # Unpack action
        steer, throttle, brake = action
        
        # Apply control
        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)
        self.vehicle.apply_control(control)
        
        # Tick the server
        if self.synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()
        
        # Calculate reward
        reward = self._get_reward()
        
        # Check termination conditions
        done = False
        if self.episode_step >= self.max_episode_steps:
            done = True
        if len(self.collision_hist) > 0:  # collision happened
            done = True
            reward -= 100.0
        
        # Get new observation
        obs = self._get_observation()
        
        # Return step information
        info = {}
        return obs, reward, done, info
    
    def _get_observation(self):
        """
        Returns the current observation. 
        For example: front camera image. 
        """
        # If you want to combine with speed or other sensor:
        #   velocity = self.vehicle.get_velocity()
        #   speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        #   ...
        return self.front_camera
    
    def _get_reward(self):
        """
        Returns the reward based on current state.
        Basic example:
          - small positive reward for moving forward
          - large negative reward for collisions
          - additional shaping for staying in lane, etc.
        """
        # Speed-based reward
        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        # Encourage some moderate speed
        speed_reward = -abs(speed - 10.0) * self.speed_weight
        
        # Additional terms could be added: lane-keeping, etc.
        reward = speed_reward
        
        return float(reward)
    
    def _on_collision(self, event):
        """Collision callback."""
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.collision_hist.append(intensity)
    
    def _process_camera(self, image):
        """Camera callback: convert raw data to RGB array."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        self.front_camera = array[:, :, :3]
        # Optionally resize or convert color space, e.g.:
        self.front_camera = cv2.cvtColor(self.front_camera, cv2.COLOR_BGR2RGB)
    
    def _destroy_actors(self):
        """Clean-up all actors (sensors, vehicle) to avoid memory leaks."""
        actors_to_destroy = [
            self.collision_sensor,
            self.camera_sensor,
            self.vehicle
        ]
        for actor in actors_to_destroy:
            if actor is not None:
                actor.destroy()
    
    def close(self):
        """Close the environment properly."""
        self._destroy_actors()
        if self.synchronous_mode:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)


# ----------------------------------------------------------------------------------
# 2) Training Script
# ----------------------------------------------------------------------------------
def main():
    # Configuration for the environment
    config = {
        "host": "localhost",
        "port": 2000,
        "timeout": 10.0,
        "synchronous_mode": False,    # set True for synchronous mode
        "max_episode_steps": 300,
        "speed_weight": 0.1
    }
    
    # Create Gym environment
    env = CarlaEnv(config)
    
    # Wrap into a DummyVecEnv (required by SB3 for single env)
    env = DummyVecEnv([lambda: env])
    
    # Create the model (PPO)
    model = PPO(
        "CnnPolicy",   # if using image-based observations
        env,
        verbose=1,
        tensorboard_log="./ppo_carla_tensorboard/"
    )
    
    # Train the model
    # Adjust total_timesteps as needed (e.g., 100000 for a decent start)
    model.learn(total_timesteps=10000)
    
    # Save the model
    model.save("ppo_carla_model")
    
    # (Optional) Load the model later
    # model = PPO.load("ppo_carla_model", env=env)
    
    # Test the trained model
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()  # optional if you implement a render() method
        if done[0]:
            obs = env.reset()


if __name__ == "__main__":
    main()
