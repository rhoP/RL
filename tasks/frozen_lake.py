from typing import DefaultDict, List, Tuple
import gymnasium as gym
from collections import defaultdict

env = gym.make(
    "FrozenLake-v1", desc=None, map_name="4x4", is_slippery=True, render_mode="human"
)
state, info = env.reset()


def generate_random_policy(env=env, episode_length=20):
    episode = []
    state, _ = env.reset()
    for _ in range(1, episode_length + 1):
        episode.append(state)
        new_action = env.action_space.sample()
        episode.append(new_action)
        state, reward, term, _, _ = env.step(new_action)
        episode.append(state)
        episode.append(reward)
        if term:
            break
    return episode


# On-policy first-visit MC control
action_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
actions = list(action_map.keys())
agent_returns: DefaultDict[Tuple[int, int], DefaultDict[int, List[float]]] = (
    defaultdict(lambda: defaultdict(list))
)


generate_random_policy()
