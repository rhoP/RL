import gymnasium as gym
import numpy as np


class CustomFrozenLake(gym.Wrapper):
    def __init__(self, env, slip_probabilities=None):
        super().__init__(env)
        self.slip_probabilities = slip_probabilities or self._default_slip_probs()

    def _default_slip_probs(self):
        """
        Default custom slippery probabilities:
        """
        return {
            0: [0.4, 0.4, 0., 0.2],  # Left: 40% left, 40% down, 0% right, 20% up
            1: [1.0 / 3, 1.0 / 3, 1.0 / 3, 0.],  # Down: 33% left, 33% down, 33% right, 0% up
            2: [0.0, 0.3, 0.4, 0.3],  # Right: 0% left, 30% down, 40% right, 30% up
            3: [1.0 / 3, 1.0 / 3, 1.0 / 3, 0.]  # Up: 33% left, 0% down, 33% right, 33% up
        }

    def step(self, action):
        # Apply custom slippery probabilities
        intended_action = action
        probs = self.slip_probabilities[intended_action]
        actual_action = np.random.choice([0, 1, 2, 3], p=probs)

        # Use the actual action but track what was intended
        obs, reward, terminated, truncated, info = self.env.step(actual_action)
        info['intended_action'] = intended_action
        info['actual_action'] = actual_action
        info['slipped'] = intended_action != actual_action

        return obs, reward, terminated, truncated, info


def create_custom_frozenlake(slip_probs=None, map_name="4x4"):
    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=True)
    return CustomFrozenLake(env, slip_probs)
