import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from carla_gym_env import CarlaGymEnv

def main():
    # 1) Create environment with a chosen render_mode.
    # - If "human", a PyGame window can appear when we call env.render().
    # - If None, no live rendering. 
    env = CarlaGymEnv(render_mode="human")  
    
    # 2) Wrap in a DummyVecEnv for SB3
    env = DummyVecEnv([lambda: env])

    # 3) Create the PPO model using a CNN policy (since we have an image observation)
    model = PPO("CnnPolicy", env, verbose=1, device="cuda")
    
    # 4) Train
    model.learn(total_timesteps=5000)

    # 5) Evaluate / test - we can call env.render() to show the PyGame window
    obs = env.reset()
    for _ in range(200):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        
        # This calls `render(mode=self.render_mode)` => "human"
        env.render()  # PyGame update

        if done:
            obs = env.reset()

    env.close()

if __name__ == "__main__":
    main()
