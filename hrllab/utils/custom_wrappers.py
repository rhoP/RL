import gymnasium as gym
from gymnasium import Wrapper
import numpy as np


class CustomRewardWrapper(Wrapper):
    """Wrapper to modify Frozen Lake rewards: -1 for all tiles, +100 for goal."""

    def __init__(self, env):
        super().__init__(env)
        self.n_states = env.observation_space.n
        self.desc = env.desc if hasattr(env, 'desc') else None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated and obs == self.n_states - 1:
            reward = 1.0

        else:
            reward = 0.0

        return obs, reward, terminated, truncated, info

    def is_final_state(self, state):
        """Check if state is the goal state."""
        return state == self.n_states - 1

    def is_hole_state(self, state):
        """Check if state is a hole."""
        if self.desc is not None:
            row = state // int(np.sqrt(self.n_states))
            col = state % int(np.sqrt(self.n_states))
            return self.desc[row][col] == b'H'
        return False


class TrackingWrapper(gym.Wrapper):
    """Wrapper that tracks state transitions for backward graph building."""

    def __init__(self, env, env_idx):
        super().__init__(env)
        self.env_idx = env_idx
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.transition_info = None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # Store transition information
        self.transition_info = {
            'env_idx': self.env_idx,
            'state': self.last_state,
            'action': self.last_action,
            'reward': self.last_reward,
            'next_state': obs,
            'done': done
        }

        # Update buffers
        self.last_state = obs
        self.last_action = action
        self.last_reward = reward

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_state = obs
        self.last_action = None
        self.last_reward = None
        self.transition_info = None
        return obs, info

    def get_transition_info(self):
        return self.transition_info
