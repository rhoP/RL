import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import argparse
import ale_py


def test(env_id, agent_id, policy):
    env = make_atari_env(
        env_id, n_envs=1, seed=69, wrapper_kwargs=dict(terminal_on_life_loss=False)
    )
    obs = env.reset()
    model = eval(agent_id)(policy, env)
    model.load(f"pre_trained/{agent_id}_{env_id}")

    # evaluate_policy(model, env, n_eval_episodes=10)
    while True:
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, info = env.step(action)
        env.render("human")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="ALE/Breakout-v5",
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

    args = parser.parse_args()

    test(args.env, args.agent, args.policy)
