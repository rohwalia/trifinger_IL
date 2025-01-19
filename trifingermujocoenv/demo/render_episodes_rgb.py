from matplotlib import pyplot as plt
import numpy as np
from trifinger_mujoco_env import TriFingerEnv


env = TriFingerEnv(render_mode="rgb_array", start_viewer=True, image_obs=True)
marker = env.add_marker(
    {
        "pos": np.array([0.0, 0.0, 0.05]),
        "rgba": [1.0, 0.0, 0.0, 1.0],
        "type": 2, # shpere
        "size": [0.02, 0.02, 0.02]
    }
)

obs, _ = env.reset()

for t in range(10):
    marker.update("pos", np.array([0.08 * np.sin(2 * np.pi * t / 10), 0.08 * np.cos(2 * np.pi * t / 10), 0.05]))
    action = env.action_space.sample()
    obs, _, _, _, info = env.step(action)
    fig, axs = plt.subplots(1, 4)
    for i, ax in enumerate(axs):
        if i < 3:
            ax.imshow(obs["camera_images"][i])
        else:
            ax.imshow(env.render())
    plt.show()


