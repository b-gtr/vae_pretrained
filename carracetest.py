# train_car1.py

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from carla_gym_env import CarlaGymEnv  # Adjust to your file name

def main():
    env = CarlaGymEnv(display=False)  # or True to see PyGame
    # Wrap in VecEnv for SB3
    env = DummyVecEnv([lambda: env])

    # Now we can use 'CnnPolicy' because we have channel-first images
    model = PPO("CnnPolicy", env, verbose=1, device='cuda')  # or 'cpu'
    model.learn(total_timesteps=10_000)

    obs = env.reset()
    for _ in range(200):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()

if __name__ == "__main__":
    main()
