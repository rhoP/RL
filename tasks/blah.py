import gymnasium
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

env = gymnasium.make(
    "FrozenLake-v1", desc=generate_random_map(size=8), render_mode="human"
)
env.reset()

for _ in range(1000):
    act = env.action_space.sample()
    n, r, t, d, i = env.step(act)
    if t or d:
        o, ii = env.reset()
        print(ii)
    print(i)
