import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DQN, A2C
import argparse
from ale_py import ALEInterface, roms
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike


def train(env_id, agent_id):
    env = gym.make(env_id)
    env.reset()
    if agent_id == "DQN" and env_id == "CartPole-v1":
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=2.3e-3,
            learning_starts=1000,
            batch_size=64,
            gamma=0.99,
            target_update_interval=10,
            train_freq=256,
            gradient_steps=128,
            exploration_fraction=0.16,
            exploration_final_eps=0.04,
        )
        model.learn(total_timesteps=50000)
    elif agent_id == "PPO" and env_id == "LunarLander-v3":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=1024,
            batch_size=64,
            gae_lambda=0.98,
            gamma=0.999,
            n_epochs=4,
            ent_coef=0.01,
        )
        model.learn(total_timesteps=1000000)
    elif agent_id == "A2C" and env_id == "ALE/Breakout-v5":
        env = make_atari_env(env_id, n_envs=16, seed=69)
        env = VecFrameStack(env, n_stack=16)
        model = A2C(
            "CnnPolicy",
            env,
            verbose=1,
            gamma=0.999,
            ent_coef=0.01,
            vf_coef=0.25,
            policy_kwargs=dict(
                optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5)
            ),
        )
        model.learn(total_timesteps=10000000)
    model.save(f"pre_trained/{agent_id}_{env_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="FrozenLake-v1",
        choices=[
            "LunarLander-v3",
            "CartPole-v1",
            "Acrobot-v1",
            "MountainCar-v0",
            "ALE/Breakout-v5",
        ],
    )
    parser.add_argument(
        "--agent", type=str, default="PPO", choices=["PPO", "A2C", "DQN"]
    )
    parser.add_argument("--episodes", type=int, default=200)

    args = parser.parse_args()

    train(args.env, args.agent)
