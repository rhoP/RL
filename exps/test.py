import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import argparse


def test(env_id, agent_id, model_path):
    env = gym.make(env_id, render_mode="human")
    obs, _ = env.reset()

    model = eval(agent_id).load(model_path)

    evaluate_policy(model, env, n_eval_episodes=10)
    # while True:
    #    action, _states = model.predict(obs, deterministic=False)
    #    obs, rewards, terminated, truncated, info = env.step(action)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="LunarLander-v3",
        choices=[
            "LunarLander-v3",
            "CartPole-v1",
            "Acrobot-v1",
            "MountainCar-v0",
            "ALE/Breakout-v5",
        ],
    )
    parser.add_argument(
        "--agent", type=str, default="A2C", choices=["PPO", "A2C", "DQN"]
    )
    parser.add_argument(
        "--policy", type=str, default="CnnPolicy", choices=["MlpPolicy", "CnnPolicy"]
    )
    parser.add_argument("--model_path", type=str, required=True)

    args = parser.parse_args()

    test(args.env, args.agent, args.model_path)
