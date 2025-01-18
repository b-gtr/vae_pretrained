# usage_example.py
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from your_carla_env_file import CarlaGymEnv  # Import the environment class

def main():
    env = CarlaGymEnv(render_mode='human')  # or 'human' for live rendering
    env = DummyVecEnv([lambda: env])        # Wrap in a VecEnv for SB3

    model = PPO("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)

    # Test the trained model
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()

if __name__ == "__main__":
    main()
