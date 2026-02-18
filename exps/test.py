import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
import argparse


def test(env_id, agent_id, policy):
    env = gym.make(env_id, render_mode="human")
    env.reset()
    model = eval(agent_id)(policy, env)
    model.load(f"pre_trained/{agent_id}_{env_id}")

    evaluate_policy(model, env, n_eval_episodes=10)


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
    parser.add_argument(
        "--policy", type=str, default="MlpPolicy", choices=["MlpPolicy", "CnnPolicy"]
    )

    args = parser.parse_args()

    test(args.env, args.agent, args.policy)
