import argparse

from matplotlib import pyplot as plt
import numpy as np

from trifinger_mujoco_env import TriFingerEnv


# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--render-mode",
    type=str,
    default="human",
    choices=["human", "rgb_array"],
)
parser.add_argument(
    "--objects",
    type=str,
    default="cube",
    choices=["cube", "two_cubes", "no_objects"],
)
args = parser.parse_args()

pos_gain = np.array([15, 15, 9] * 3)


env = TriFingerEnv(render_mode=args.render_mode, objects=args.objects)

obs, _ = env.reset()
desired_joint_angles = None

desired_tip_pos = np.array(
    [
        [0.0, 0.12, 0.09],
        [0.05, 0.0, 0.05],
        [-0.05, 0.0, 0.05],
    ]
)
colors = [
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
]

for tip_pos, color in zip(desired_tip_pos, colors):
    env.add_marker(
        {
            "pos": tip_pos,
            "rgba": color,
            "type": 2, # shpere
            "size": [0.01, 0.01, 0.01]
        }
    )

from time import time
for i in range(3000):
    if i == 0:
        action = np.zeros(9)
    else:
        action = pos_gain * (desired_joint_angles.reshape(9) - obs["robot"]["position"][:9])
    obs, _, _, _, info = env.step(action)
    tip_pos = env.get_tip_position()
    time0 = time()
    desired_joint_angles = env.get_angles_from_tip_pos(desired_tip_pos)
    print(time()-time0)
    if args.render_mode == "rgb_array":
        obs, image = env.get_observation()
        plt.imshow(image)
        plt.show()