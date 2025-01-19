import numpy as np

from trifinger_mujoco_env import TriFingerEnv


env = TriFingerEnv(render_mode="human")
marker = env.add_marker({"pos": np.array([0.0, 0.0, 0.06]), "rgba": [1.0, 0.0, 0.0, 1.0], "type": 2, "size": [0.02, 0.02, 0.02]})

obs, _ = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, _, _, _, _ = env.step(action)