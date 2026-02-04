import minari
import gymnasium as gym
from minari import DataCollector

env = gym.make("CartPole-v1")
env = DataCollector(env, record_infos=True)

total_episodes = 100

for _ in range(total_episodes):
    env.reset(seed=123)
    while True:
        # random action policy
        action = env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

dataset = env.create_dataset(
    dataset_id="cartpole/test-v0",
    algorithm_name="Random-Policy",
)
