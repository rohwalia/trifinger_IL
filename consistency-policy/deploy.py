import torch
from torch import nn
from diffusion_policy.dataset.base_dataset import BaseImageDataset, LinearNormalizer
from consistency_policy.utils import get_policy, rmat_to_quat, rot6d_to_rmat
from consistency_policy.policy_wrapper import PolicyWrapper
import numpy as np
from trifinger_mujoco_env import MoveCubeEnv, TriFingerEnv
from trifinger_mujoco_env.utils import get_keypoints_from_pose
import os
import numpy as np
from matplotlib import pyplot as plt
import quaternion
import cv2
import mujoco
import copy


def get_pos_from_angle(target_angle, env):
    kinematic_data = mujoco.MjData(env.model)
    kinematic_data.qpos[:9] = target_angle
    mujoco.mj_kinematics(env.model, kinematic_data)
    tip_positions = np.zeros((3, 3), dtype=np.float32)
    for i, alpha in enumerate([0, 120, 240]):
        tip_positions[i] = kinematic_data.geom(f"tip_{alpha}").xpos
    return tip_positions


task = "push"
frequency = 10
dt = 0.005
#env = TriFingerEnv(objects="no_objects", render_mode="rgb_array", dt=0.005, n_substeps=2, image_width=480, image_height=270, image_obs=True)
env = MoveCubeEnv(task=task, fix_task=True, render_mode=None, dt=dt, n_substeps=2, image_width=480, image_height=270, image_obs=True)
env.reset()
#env_view = TriFingerEnv(objects="no_objects", render_mode="rgb_array", dt=0.005, n_substeps=2, image_width=480, image_height=270, image_obs=True)
env_view = MoveCubeEnv(task=task, fix_task=True, render_mode="rgb_array", dt=dt, n_substeps=2, image_width=480, image_height=270, image_obs=True)
env_view.set_goal(env.goal_position, env.goal_orientation)
marker_thumb = env_view.add_marker(
    {
        "pos": np.array([0.0, 0.0, 0.1]),
        "rgba": [1.0, 0.0, 0.0, 1.0],
        "type": 2, # shpere
        "size": [0.005, 0.005, 0.005]
    },
    visible_to_cameras=True,
)
marker_index = env_view.add_marker(
    {
        "pos": np.array([0.0, 0.0, 0.1]),
        "rgba": [0.0, 1.0, 0.0, 1.0],
        "type": 2, # shpere
        "size": [0.005, 0.005, 0.005]
    },
    visible_to_cameras=True,
)
marker_middle = env_view.add_marker(
    {
        "pos": np.array([0.0, 0.0, 0.1]),
        "rgba": [0.0, 0.0, 1.0, 1.0],
        "type": 2, # shpere
        "size": [0.005, 0.005, 0.005]
    },
    visible_to_cameras=True,
)

ckpt_path = "latest_push.ckpt"

policy = get_policy(ckpt_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
policy = policy.to(device)
policy.eval()
for param in policy.parameters():
    param.requires_grad = False

pw = PolicyWrapper(policy, n_obs=2, n_acts=8, d_pos=6, d_rot=6, device=device)

episode_length = 100
p_gain = np.array([2.5, 2.5, 1.5] * 3)
d_gain = np.array([0.1, 0.2, 0.1] * 3)
#camera_images = np.full((episode_length, 3, 270, 360, 3), 0.0, dtype=np.uint8)
images = []
reward = []
scale_height, scale_width = 135, 180
for i in range(episode_length):
    camera_images = np.array([cv2.resize(image, (scale_width, scale_height), interpolation=cv2.INTER_LINEAR) for image in env.get_camera_images()])
    obs = env.get_observation(detail = False, state=True)
    input = {
    'camera_0': np.array(camera_images[0], dtype=np.uint8),
    'camera_1': np.array(camera_images[1], dtype=np.uint8),
    'camera_2': np.array(camera_images[2], dtype=np.uint8),
    'robot_pose': np.array(obs["robot"]["position"]),
    'torques': np.array(obs["sim_state"]["ctrl"])
    #'cube_keypoints': get_keypoints_from_pose(obs["sim_state"]["qpos"][9:12], quaternion.from_float_array(obs["sim_state"]["qpos"][12:16])).flatten()
    }
    if i==0:
        fig, axs = plt.subplots(1, 4)
        for j, ax in enumerate(axs):
            if j < 3:
                ax.imshow(np.array(camera_images[j], dtype=np.uint8))
        plt.show()
    output = pw.get_action(input)
    target_pos = get_pos_from_angle(output, env)
    marker_thumb.update("pos", target_pos[0])
    marker_index.update("pos", target_pos[1])
    marker_middle.update("pos", target_pos[2])
    env_view.data = copy.deepcopy(env.data)
    images.append(env_view.render()[:, :, ::-1].astype(np.uint8))
    obs["action"] = output
    obs_temp = obs
    for j in range(int(1/(frequency*dt*2))):
        delta = output - obs_temp["robot"]["position"]
        action = p_gain * delta - d_gain * obs_temp["robot"]["velocity"] #(delta-self.prev_delta)/0.02
        _, rew, _, _, has_achieved = env.step(action, return_obs=False)
        obs_temp = env.get_observation(detail = False, state=True)
        if j==int(1/(frequency*dt*2))-1:
            obs["rew"] = rew
            obs["has_achieved"] = has_achieved
    print(i)
del env
del env_view
# Load images from the specified camera key
images = np.array(images, dtype=np.uint8)

# Define frame properties
height, width, _ = images.shape[1:]

# Specify the output file and codec (e.g., 'mp4v' for .mp4 files)
output_file = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Initialize the video writer object
video_writer = cv2.VideoWriter(output_file, fourcc, frequency, (width, height))

# Iterate over images and write them to the video file
for image in images:
    video_writer.write(image)

# Release the video writer object
video_writer.release()

print(f"Video saved to {output_file}")
