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


task = "lift"
#real_task = "".
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

ckpt_path = "latest_lift.ckpt"

policy = get_policy(ckpt_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
policy = policy.to(device)
policy.eval()
for param in policy.parameters():
    param.requires_grad = False

pw = PolicyWrapper(policy, n_obs=2, n_acts=8, d_pos=6, d_rot=6, device=device)

runs = 30
episode_length = 150
p_gain = np.array([2.5, 2.5, 1.5] * 3)
d_gain = np.array([0.1, 0.2, 0.1] * 3)
reward = []
has_achieved_status = []
scale_height, scale_width = 135, 180
for run in range(runs):
    images = []
    reward_temp = []
    has_achieved_status_temp = []
    for i in range(episode_length):
        camera_images = np.array([cv2.resize(image, (scale_width, scale_height), interpolation=cv2.INTER_LINEAR) for image in env.get_camera_images()])
        obs = env.get_observation(detail = False, state=True)
        input = {
        'camera_0': np.array(camera_images[0], dtype=np.uint8),
        'camera_1': np.array(camera_images[1], dtype=np.uint8),
        'camera_2': np.array(camera_images[2], dtype=np.uint8),
        'robot_pose': np.array(obs["robot"]["position"]),
        'torques': np.array(obs["sim_state"]["ctrl"])
        }
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
            if j==int(1/(frequency*dt*2))-1:
                _, obs["rew"], _, _, obs["has_achieved"] = env.step(action, return_obs=True)
            else:
                _, _, _, _, _ = env.step(action, return_obs=False)
        reward_temp.append(obs["rew"])
        has_achieved_status_temp.append(obs["has_achieved"])
        # if obs["has_achieved"] == 1:
        #     break
    images = np.array(images, dtype=np.uint8)
    height, width, _ = images.shape[1:]
    output_file = 'eval/{}_teacher/video/run_{}.mp4'.format(task, run)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, frequency, (width, height))
    for image in images:
        video_writer.write(image)
    video_writer.release()
    print(f"Video saved to {output_file}")
    reward.append(reward_temp)
    has_achieved_status.append(has_achieved_status_temp)
    env.reset()
reward = np.array(reward)
has_achieved_status = np.array(has_achieved_status)
np.save('eval/{}_teacher/has_achieved.npy'.format(task), has_achieved_status)
np.save('eval/{}_teacher/reward.npy'.format(task), reward)
print(np.any(has_achieved_status[:,], axis=1))
print(np.mean(np.any(has_achieved_status[:,], axis=1))) #has_achieved_status.shape[1] // 2: