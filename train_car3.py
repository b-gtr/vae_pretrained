# train_car.py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from carla_gym_env import CarlaGymEnv

def main():
    # Erstelle Env mit z.B. 2000 Schritten max.
    env = CarlaGymEnv(display=False, max_steps=2000)
    env = DummyVecEnv([lambda: env])

    model = PPO("MultiInputPolicy", env, verbose=1, device="cuda")
    model.learn(total_timesteps=20000)

    # Testlauf
    obs = env.reset()
    for _ in range(500):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()

    env.close()

if __name__ == "__main__":
    main()
