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
import h5py


def get_pos_from_angle(target_angle, env):
    kinematic_data = mujoco.MjData(env.model)
    kinematic_data.qpos[:9] = target_angle
    mujoco.mj_kinematics(env.model, kinematic_data)
    tip_positions = np.zeros((3, 3), dtype=np.float32)
    for i, alpha in enumerate([0, 120, 240]):
        tip_positions[i] = kinematic_data.geom(f"tip_{alpha}").xpos
    return tip_positions


def get_train_traj(hdf5_path, demo_num):
    with open(os.path.splitext(hdf5_path)[0] + '.txt', 'r') as file:
        file_list = file.read().splitlines()
    original_file = file_list[demo_num-1]
    # Open the HDF5 file
    with h5py.File(hdf5_path, 'r') as hdf_file:
        if demo_num is not None:
            group_name = f'demo_{demo_num}'
            if group_name in hdf_file:
                group = hdf_file[group_name]
                observations = [group['camera_0'][:], group['camera_1'][:], group['camera_2'][:], group['robot_pose'][:], group['torques'][:]]
                actions = group['action'][:]
    with h5py.File(original_file, 'r') as hdf_file:
        states = [hdf_file['sim_state/qpos'][:], hdf_file['sim_state/qvel'][:], hdf_file['sim_state/ctrl'][:]]
        goal = [hdf_file["info/goal_position"][:], hdf_file["info/goal_orientation"][:]]
    return observations, actions, states, goal

def get_mse_error(demo_num, pw):
    pw.reset()
    hdf5_path = "/home/local_rohan/consistency-policy/data/push/push.hdf5"
    observations, actions, states, goal = get_train_traj(hdf5_path, demo_num)
    pred_actions = []
    for i in range(len(observations[0])):
        input = {
        'camera_0': np.array(observations[0][i], dtype=np.uint8),
        'camera_1': np.array(observations[1][i], dtype=np.uint8),
        'camera_2': np.array(observations[2][i], dtype=np.uint8),
        'robot_pose': np.array(observations[3][i]),
        'torques': np.array(observations[4][i])
        }
        pred_actions.append(pw.get_action(input))

    pred_actions = np.array(pred_actions)
    episode_length = len(observations[0])

    diff = []

    time_stamps = np.arange(episode_length)
    plt.plot(states[0][:,0], label="actual", color='red')
    plt.plot(actions[:,0], label="desired", color='green')
    plt.plot(time_stamps, pred_actions[:, :, 0][:, 0], label="predicted", linestyle='-', color='blue', linewidth=2)
    for t in range(episode_length-8):
        mse_loss = torch.nn.MSELoss()
        mse = mse_loss(torch.tensor(pred_actions[t]), torch.tensor(actions[t:t+8]))
        diff.append(mse.item())
        plt.plot(time_stamps[t:t+8], pred_actions[:, :, 0][t], 
                    linestyle='--', color='orange', linewidth=1, alpha=0.7)
    # plt.plot(exec_actions[:,0], label="executed")
    plt.legend()
    plt.savefig(os.path.join("mse_plots/", f'demo_{demo_num}.png'))
    plt.clf()
    diff = np.array(diff)
    mse_total = np.mean(diff)
    return mse_total


ckpt_path = "epoch=0036-val_loss=0.035.ckpt"

policy = get_policy(ckpt_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
policy = policy.to(device)
policy.eval()
for param in policy.parameters():
    param.requires_grad = False

pw = PolicyWrapper(policy, n_obs=2, n_acts=8, d_pos=6, d_rot=6, device=device)
errors = []
for num in np.random.randint(1, 151, size=15):
    errors.append(get_mse_error(num, pw))
print(errors)
print(np.mean(errors))


