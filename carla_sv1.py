#!/usr/bin/env python

import gym
import numpy as np
import carla
import random
import math

from gym import spaces
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# ----------------------------------------------------------------------------------
# Helper Function: Lane offset & heading error
# ----------------------------------------------------------------------------------
def get_lane_center_offset_and_heading(world_map, vehicle_transform):
    """
    Calculate how far the vehicle is from lane center (lateral offset)
    and the heading error (difference in orientation to the road center).

    Returns:
      lane_offset (float): Lateral distance to lane center [m].
      heading_error (float): Orientation difference from lane heading [radians].
    """
    waypoint = world_map.get_waypoint(
        vehicle_transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    if waypoint is None:
        return 0.0, 0.0

    lane_transform = waypoint.transform

    # Vehicle location & orientation
    vx, vy = vehicle_transform.location.x, vehicle_transform.location.y
    vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)

    # Lane center location & orientation
    lx, ly = lane_transform.location.x, lane_transform.location.y
    lane_yaw = math.radians(lane_transform.rotation.yaw)

    # Right vector of lane center
    right_vec = lane_transform.get_right_vector()
    # Vector from lane center to vehicle
    dx, dy = (vx - lx), (vy - ly)
    # Lateral offset is the projection onto the lane's right vector
    lane_offset = dx * right_vec.x + dy * right_vec.y

    # Heading error in [-pi, pi]
    heading_error = vehicle_yaw - lane_yaw
    heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

    return lane_offset, heading_error


def is_in_front(base_transform, check_transform):
    """
    Check if 'check_transform' is in front of 'base_transform' 
    based on a forward vector dot product.
    - We consider 'in front' if dot((check_loc - base_loc), forward_vector) > 0.
    """
    base_loc = base_transform.location
    base_forward = base_transform.get_forward_vector()
    check_loc = check_transform.location

    dx = check_loc.x - base_loc.x
    dy = check_loc.y - base_loc.y
    dz = check_loc.z - base_loc.z

    dot = dx * base_forward.x + dy * base_forward.y + dz * base_forward.z
    return dot > 0  # True if in front, False if behind or orthogonal


# ----------------------------------------------------------------------------------
# Custom CARLA Gym Environment
# ----------------------------------------------------------------------------------
class CarlaEnv(gym.Env):
    """
    A custom environment for CARLA focusing on lane-keeping (vector obs).
    Observations: [speed, lane_offset, heading_error]
    Actions: [steer, throttle, brake]
    Rewards: 
     - Keep lane offset small
     - Heading alignment
     - Target speed
     - Large negative for collision
    """
    def __init__(self, config):
        super(CarlaEnv, self).__init__()
        
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 2000)
        self.timeout = config.get("timeout", 10.0)
        self.target_speed = config.get("target_speed", 12.0)  # 12 m/s ~ 43 km/h
        self.max_episode_steps = config.get("max_episode_steps", 1000)

        # Connect to CARLA
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()
        self.map = self.world.get_map()

        # Synchronous mode if wanted
        self.synchronous_mode = config.get("synchronous_mode", True)
        if self.synchronous_mode:
            settings = self.world.get_settings()
            settings.fixed_delta_seconds = 0.05  # 20 FPS
            settings.synchronous_mode = True
            self.world.apply_settings(settings)
        
        self.bp_library = self.world.get_blueprint_library()

        # Observation space: [speed, lane_offset, heading_error]
        # Bound them reasonably, e.g. speed up to 40 m/s, lane offset ±5, heading_error ±π
        low_obs = np.array([0.0, -5.0, -np.pi], dtype=np.float32)
        high_obs = np.array([40.0, 5.0, np.pi], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, shape=(3,), dtype=np.float32)

        # Action space: [steer, throttle, brake]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            shape=(3,),
            dtype=np.float32
        )

        # Placeholder for actors
        self.vehicle = None
        self.collision_sensor = None
        self.collision_hist = []

        self.episode_step = 0

        # We define a reference transform for spawn filtering in front
        # For example, let's define a pseudo "base" transform in the map center or random
        # You can also define a static transform or the player's transform if known
        # For demonstration, we'll pick the first spawn point as a reference
        spawn_points = self.map.get_spawn_points()
        if len(spawn_points) == 0:
            raise ValueError("No spawn points available in the CARLA map!")
        self.reference_transform = spawn_points[0]  # or any suitable reference

    def reset(self):
        # Clean up old actors
        self._destroy_actors()
        self.episode_step = 0
        self.collision_hist.clear()

        # Spawn vehicle in a random "valid" spawn point that is in front of self.reference_transform
        spawn_points = self.map.get_spawn_points()
        valid_spawn_points = [
            sp for sp in spawn_points 
            if is_in_front(self.reference_transform, sp)
        ]
        if not valid_spawn_points:
            # Fallback: if no valid spawn points are "in front", just pick random
            valid_spawn_points = spawn_points

        spawn_point = random.choice(valid_spawn_points)
        vehicle_bp = self.bp_library.find('vehicle.tesla.model3')
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)

        # Collision sensor
        col_bp = self.bp_library.find('sensor.other.collision')
        col_sensor_transform = carla.Transform()
        self.collision_sensor = self.world.spawn_actor(col_bp, col_sensor_transform, attach_to=self.vehicle)
        self.collision_sensor.listen(lambda e: self._on_collision(e))

        # Give the simulator a few ticks to stabilize
        for _ in range(10):
            if self.synchronous_mode:
                self.world.tick()
            else:
                self.world.wait_for_tick()

        return self._get_observation()

    def step(self, action):
        self.episode_step += 1

        # Apply action
        steer, throttle, brake = action
        control = carla.VehicleControl(
            steer=float(steer),
            throttle=float(throttle),
            brake=float(brake)
        )
        self.vehicle.apply_control(control)

        # Advance simulation
        if self.synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # Compute reward
        reward = self._get_reward()

        # Check termination
        done = False
        if self.episode_step >= self.max_episode_steps:
            done = True
        if len(self.collision_hist) > 0:
            reward -= 100.0  # large penalty for collision
            done = True

        obs = self._get_observation()
        return obs, reward, done, {}

    def _get_observation(self):
        """
        Observation: [speed, lane_offset, heading_error]
        """
        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        transform = self.vehicle.get_transform()
        lane_offset, heading_error = get_lane_center_offset_and_heading(self.map, transform)

        return np.array([speed, lane_offset, heading_error], dtype=np.float32)

    def _get_reward(self):
        """
        Reward function focusing on:
          - minimal lane offset
          - minimal heading error
          - hitting target speed
        """
        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        transform = self.vehicle.get_transform()
        lane_offset, heading_error = get_lane_center_offset_and_heading(self.map, transform)

        # 1) Speed reward: penalize squared difference from target
        speed_error = speed - self.target_speed
        speed_reward = -0.05 * (speed_error ** 2)

        # 2) Lane offset penalty
        lane_offset_pen = -1.0 * abs(lane_offset)

        # 3) Heading error penalty
        heading_pen = -0.5 * abs(heading_error)

        return float(speed_reward + lane_offset_pen + heading_pen)

    def _on_collision(self, event):
        # log collision impulses
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.collision_hist.append(intensity)

    def _destroy_actors(self):
        if self.collision_sensor:
            self.collision_sensor.destroy()
            self.collision_sensor = None
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


# ----------------------------------------------------------------------------------
# Training Script with Recurrent PPO (LSTM) from sb3_contrib
# ----------------------------------------------------------------------------------
def main():
    config = {
        "host": "localhost",
        "port": 2000,
        "timeout": 10.0,
        "synchronous_mode": True,
        "target_speed": 12.0,
        "max_episode_steps": 500
    }

    # Create and vectorize environment
    env = DummyVecEnv([lambda: CarlaEnv(config)])
    
    # Create an evaluation environment for the callback (optional)
    eval_env = DummyVecEnv([lambda: CarlaEnv(config)])

    # Optional: an EvalCallback to track performance
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./logs_recurrent/',
        log_path='./logs_recurrent/',
        eval_freq=20_000,
        deterministic=True,
        render=False
    )

    # Use RecurrentPPO (LSTM policy) from sb3-contrib
    # MlpLstmPolicy is a recurrent policy with LSTM layers
    model = RecurrentPPO(
        policy=MlpLstmPolicy,
        env=env,
        n_steps=128,            # shorter rollout length for RNN
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        clip_range=0.2,
        n_epochs=10,
        verbose=1,
        tensorboard_log="./recurrent_ppo_carla_tensorboard/"
    )

    # Train for 500,000 timesteps
    model.learn(
        total_timesteps=500000,
        callback=eval_callback
    )
    
    # Save final model
    model.save("recurrent_ppo_carla_lane_center")
    print("Training finished. Model saved as recurrent_ppo_carla_lane_center.zip")

    # (Optional) Testing / inference
    # model = RecurrentPPO.load("recurrent_ppo_carla_lane_center", env=env)
    # obs = env.reset()
    # lstm_states = None
    # episode_starts = np.ones((env.num_envs,), dtype=bool)
    # for _ in range(1000):
    #     action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     episode_starts = done
    #     if done[0]:
    #         obs = env.reset()
    #         lstm_states = None
    #         episode_starts = np.ones((env.num_envs,), dtype=bool)


if __name__ == "__main__":
    main()
