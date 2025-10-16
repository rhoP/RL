import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch as th
import multiprocessing as mp
import warnings
from hrllab.utils.BackwardGraphBuilder import BackwardGraphBuilder
from hrllab.utils.callbacks import TrackingWrapperCallback
from hrllab.utils.custom_wrappers import CustomRewardWrapper
from hrllab.utils.Env import make_tracking_env

warnings.filterwarnings('ignore')


def main():
    env = TimeLimit(CustomRewardWrapper(gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")),
                    max_episode_steps=100)

    n_envs = 8

    seed = 791

    backward_builder = BackwardGraphBuilder(n_envs)

    env_fns = [make_tracking_env(i) for i in range(n_envs)]
    vec_env = VecNormalize(DummyVecEnv(env_fns), norm_obs=False)
    vec_env.seed(seed)

    callback = TrackingWrapperCallback(backward_builder, save_freq=100)

    policy_kwargs = dict(
        activation_fn=th.nn.ReLU,
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=0.001,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device='cpu',
        seed=seed,
    )

    print("Training model with parallel environments...")
    model.learn(total_timesteps=70000, callback=callback, progress_bar=True)

    backward_fig = backward_builder.plot_backward_graph()
    if backward_fig:
        plt.savefig('plots/backward_graph.png', dpi=300, bbox_inches='tight')
        plt.plot()

    # Test the final policy
    print("Testing final policy:")

    succ_counter = 0
    for j in range(10):
        print(f"Iteration: {j}")
        obs, _ = env.reset()
        for i in range(25):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            print(f"\tStep {i}: State={obs}, Action={action}, Reward={reward}, Terminated={terminated}")
            if truncated:
                break
            elif terminated and int(obs) == 15:
                succ_counter += 1
                break
            elif terminated:
                break
    print(f"Final policy has {succ_counter} successful episodes.")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
