import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, episode_len, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.episode_len = episode_len
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        with h5py.File(self.dataset_dir, 'r') as hdf_file:
            is_sim = True
            original_action_shape = hdf_file["demo_{}".format(episode_id)]['action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            robot_pose = hdf_file["demo_{}".format(episode_id)]['robot_pose'][start_ts]
            torque = hdf_file["demo_{}".format(episode_id)]['torques'][start_ts]
            qpos = np.hstack((robot_pose, torque))
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = hdf_file["demo_{}".format(episode_id)][f'{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = hdf_file["demo_{}".format(episode_id)]['action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = hdf_file["demo_{}".format(episode_id)]['action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros((self.episode_len, original_action_shape[1]), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    lengths = []

    # Read data from the HDF5 file
    with h5py.File(dataset_dir, 'r') as hdf_file:
        for i in hdf_file.keys():
            robot_pose = hdf_file[i]['robot_pose'][:]
            torque = hdf_file[i]['torques'][:]
            qpos = np.hstack((robot_pose, torque))
            action = hdf_file[i]['action'][:]
            all_qpos_data.extend(qpos)
            all_action_data.extend(action)
            lengths.append(len(robot_pose))

    # Stack the data to form arrays
    all_qpos_data = np.array(all_qpos_data)
    all_action_data = np.array(all_action_data)

    # Normalize action data
    action_mean = np.mean(all_action_data, axis=(0, 1), keepdims=True)
    action_std = np.std(all_action_data, axis=(0, 1), keepdims=True)
    action_std = np.clip(action_std, 1e-2, np.inf)  # Clipping

    # Normalize qpos data
    qpos_mean = np.mean(all_qpos_data, axis=(0, 1), keepdims=True)
    qpos_std = np.std(all_qpos_data, axis=(0, 1), keepdims=True)
    qpos_std = np.clip(qpos_std, 1e-2, np.inf)  # Clipping

    # Prepare stats dictionary
    stats = {
        "action_mean": action_mean.squeeze(),
        "action_std": action_std.squeeze(),
        "qpos_mean": qpos_mean.squeeze(),
        "qpos_std": qpos_std.squeeze(),
        "example_qpos": qpos
    }

    return stats


def load_data(dataset_dir, num_episodes, episode_len, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes) + 1
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)
    print(norm_stats)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, episode_len, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, episode_len, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
