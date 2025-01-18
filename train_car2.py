# train_car.py
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from carla_gym_env import CarlaGymEnv

def main():
    env = CarlaGymEnv(display=True)
    # In Stable Baselines 3 müssen wir das Env vectorisieren:
    env = DummyVecEnv([lambda: env])

    # PPO-Instanz
    model = PPO("MultiInputPolicy", env, verbose=1, device="cuda")  # oder "cpu"

    model.learn(total_timesteps=10000)

    obs = env.reset()
    for _ in range(200):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()

    env.close()

if __name__ == "__main__":
    main()
