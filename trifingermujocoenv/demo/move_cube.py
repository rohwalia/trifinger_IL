import numpy as np

from trifinger_mujoco_env import MoveCubeEnv


env = MoveCubeEnv(task="push_reorient", render_mode="human")

obs, _ = env.reset()

for _ in range(3000):
    action = env.action_space.sample()
    obs, rew, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()