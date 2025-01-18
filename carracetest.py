# train_car1.py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from carla_gym_env import CarlaGymEnv  # Adjust import to your file name

def main():
    env = CarlaGymEnv(display=False)
    # Wrap in VecEnv for SB3
    env = DummyVecEnv([lambda: env])

    model = PPO("CnnPolicy", env, verbose=1, device="cuda")  # or "cpu"
    model.learn(total_timesteps=10000)

    obs = env.reset()
    for _ in range(200):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()  # show PyGame if display=True in the env
        if done:
            obs = env.reset()
    env.close()

if __name__ == "__main__":
    main()
