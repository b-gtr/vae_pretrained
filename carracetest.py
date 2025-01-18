# usage_example.py
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from carla_gym_env import CarlaGymEnv  # Adjust import to your file name

def main():
    # Create environment
    env = CarlaGymEnv(display=True)  # or False if you don't want a PyGame window
    env = DummyVecEnv([lambda: env])  # Wrap in VecEnv for SB3

    # Create and train model
    model = PPO("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)

    # Test the trained model
    obs = env.reset()
    for _ in range(200):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()  # If display=True
        if done:
            obs = env.reset()

    env.close()

if __name__ == "__main__":
    main()
