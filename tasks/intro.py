import gymnasium as gym
from stable_baselines3 import A2C
import os


models_dir = "../models/A2C"
logdir = "../logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)


if not os.path.exists(logdir):
    os.makedirs(logdir)


env = gym.make("LunarLander-v3", render_mode="human")
env.reset()

model = A2C("MlpPolicy", env, verbose=1)
time_steps = 10000
for i in range(30):
    model.learn(
        total_timesteps=time_steps, reset_num_timesteps=False, tb_log_name="A2C"
    )
    model.save(f"{models_dir}/{time_steps * i}")

"""episodes = 10

for ep in range(episodes):
    state = env.reset()
    done = False
    while not done:
        for step in range(200):
            env.render()
            obs, reward, done, info, _ = env.step(env.action_space.sample())
            print(reward)"""


env.close()
