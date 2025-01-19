import torch
from torch import nn
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
from utils import set_seed
from einops import rearrange
from policy import ACTPolicy, CNNMLPPolicy
import pickle

def get_pos_from_angle(target_angle, env):
    kinematic_data = mujoco.MjData(env.model)
    kinematic_data.qpos[:9] = target_angle
    mujoco.mj_kinematics(env.model, kinematic_data)
    tip_positions = np.zeros((3, 3), dtype=np.float32)
    for i, alpha in enumerate([0, 120, 240]):
        tip_positions[i] = kinematic_data.geom(f"tip_{alpha}").xpos
    return tip_positions

def get_image(obs, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(obs[cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy



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

camera_names = ['camera_0', 'camera_1', 'camera_2']
kl_weight = 10
lr = 1e-5
chunk_size = 20
hidden_dim = 512 
dim_feedforward = 3200
policy_class = "ACT"
state_dim = 18
action_dim = 9
lr_backbone = 1e-5
backbone = 'resnet18'
if policy_class == 'ACT':
    enc_layers = 4
    dec_layers = 7
    nheads = 8
    policy_config = {'lr': lr,
                        'num_queries': chunk_size,
                        'kl_weight': kl_weight,
                        'hidden_dim': hidden_dim,
                        'dim_feedforward': dim_feedforward,
                        'lr_backbone': lr_backbone,
                        'backbone': backbone,
                        'enc_layers': enc_layers,
                        'dec_layers': dec_layers,
                        'nheads': nheads,
                        'camera_names': camera_names,
                        }
elif policy_class == 'CNNMLP':
    policy_config = {'lr': lr, 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                        'camera_names': camera_names,}
else:
    raise NotImplementedError

set_seed(1000)
ckpt_dir = "/home/local_rohan/act/checkpoints/lift"
ckpt_name = "policy_best.ckpt"
temporal_agg = True
ckpt_path = os.path.join(ckpt_dir, ckpt_name)
policy = make_policy(policy_class, policy_config)
loading_status = policy.load_state_dict(torch.load(ckpt_path))
print(loading_status)
policy.cuda()
policy.eval()
print(f'Loaded: {ckpt_path}')
stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
with open(stats_path, 'rb') as f:
    stats = pickle.load(f)

pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
post_process = lambda a: a * stats['action_std'] + stats['action_mean']

query_frequency = policy_config['num_queries']
if temporal_agg:
    query_frequency = 1
    num_queries = policy_config['num_queries']

runs = 80
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
    if temporal_agg:
        all_time_actions = torch.zeros([episode_length, episode_length+num_queries, action_dim]).cuda()
    qpos_history = torch.zeros((1, episode_length, state_dim)).cuda()
    with torch.inference_mode():
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
            qpos_numpy = np.hstack((np.array(obs["robot"]["position"]), np.array(obs["sim_state"]["ctrl"])))
            qpos = pre_process(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            qpos_history[:, i] = qpos
            curr_image = get_image(input, camera_names)
            if policy_class == "ACT":
                if i % query_frequency == 0:
                    all_actions = policy(qpos, curr_image)
                if temporal_agg:
                    all_time_actions[[i], i:i+num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, i]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, i % query_frequency]
            elif policy_class == "CNNMLP":
                raw_action = policy(qpos, curr_image)
            else:
                raise NotImplementedError

            ### post-process actions
            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = post_process(raw_action)
            output = action

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
    output_file = 'eval/{}/{}/video/run_{}.mp4'.format(policy_class, task, run)
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
np.save('eval/{}/{}/has_achieved.npy'.format(policy_class, task), has_achieved_status)
np.save('eval/{}/{}/reward.npy'.format(policy_class, task), reward)
print(np.any(has_achieved_status[:,], axis=1))
print(np.mean(np.any(has_achieved_status[:,], axis=1))) #has_achieved_status.shape[1] // 2: