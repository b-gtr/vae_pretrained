# train_car1.py
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from carla_gym_env import CarlaGymEnv  # or whatever your file is named

def main():
    # Create environment; use display=False to avoid opening PyGame
    env = CarlaGymEnv(display=True)

    # Wrap in a VecEnv for SB3
    env = DummyVecEnv([lambda: env])

    # Create PPO model with CnnPolicy
    # This works now that observation shape is (480,640,1) channel-last
    model = PPO("CnnPolicy", env, verbose=1, device="cuda")  # or "cpu"
    model.learn(total_timesteps=10_000)

    # Test the trained model
    obs = env.reset()
    for _ in range(200):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()  # Show PyGame window if display=True
        if done:
            obs = env.reset()

    env.close()

if __name__ == "__main__":
    main()
