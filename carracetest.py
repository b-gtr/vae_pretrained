import gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from ppo4 import PPO

def make_env(env_id):
    def _init():
        env = gym.make(env_id, continuous=True)
        return env
    return _init

def main():
    env_fns = [make_env("CarRacing-v2") for _ in range(2)]
    vec_env = SubprocVecEnv(env_fns)
    
    obs = vec_env.reset()
    action_dim = vec_env.action_space.shape[0]
    model = PPO(
        in_channels=3,
        dummy_img_height=96,
        dummy_img_width=96,
        action_dim=action_dim,
        init_learning_rate=1e-4,
        lr_konstant=0.1,
        n_maxsteps=100_000,
        roullout=2048,
        n_epochs=10,
        n_envs=2,
        device="cpu"
    )

    model.collect_rollouts(vec_env)

if __name__ == "__main__":
    main()
