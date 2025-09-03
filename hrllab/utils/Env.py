import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from hrllab.utils.custom_wrappers import CustomRewardWrapper, TrackingWrapper


def make_tracking_env(env_idx):
    """Create environment with tracking wrapper."""

    def _init():
        base_env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
        wrapped_env = CustomRewardWrapper(base_env)
        tracking_env = TrackingWrapper(wrapped_env, env_idx)
        return tracking_env

    return _init


def make_env():
    """Create a function that returns a new environment instance."""
    env = gym.make('FrozenLake-v1', is_slippery=True)
    env = TimeLimit(env, max_episode_steps=100)
    return CustomRewardWrapper(env)
