import h5py
import cv2
import numpy as np
import re
import argparse
import os
import matplotlib.pyplot as plt
from trifinger_mujoco_env import MoveCubeEnv, Marker
import mujoco

def get_pos_from_angle(target_angle, env, tip):
    kinematic_data = mujoco.MjData(env.model)
    kinematic_data.qpos[:9] = target_angle
    mujoco.mj_kinematics(env.model, kinematic_data)
    tip_positions = np.zeros(3, dtype=np.float32)
    tip_positions = kinematic_data.geom(f"tip_{tip}").xpos
    return tip_positions

def find_max_demo_number(base_group):
    max_number = -1
    pattern = re.compile(r'^demo_(\d+)$')
    
    for name in base_group.keys():
        match = pattern.match(name)
        if match:
            number = int(match.group(1))
            if number > max_number:
                max_number = number
    
    return max_number

def play_video_from_original(original_file, task, fps):
    if "_low" in task:
        task = task.replace("_low", "")
    if "_fix" in task:
        task = task.replace("_fix", "")
    env = MoveCubeEnv(task=task, render_mode="rgb_array", dt=0.005, n_substeps=2, image_width=480, image_height=270, image_obs=True)
    action_steps = 8
    markers_thumb = np.empty(action_steps, dtype=Marker)
    gradient_thumb = np.linspace([1, 0, 0, 1], [1, 0.9, 0.9, 1], action_steps)
    markers_index = np.empty(action_steps, dtype=Marker)
    gradient_index = np.linspace([0, 1, 0, 1], [0.9, 1, 0.9, 1], action_steps)
    markers_middle = np.empty(action_steps, dtype=Marker)
    gradient_middle = np.linspace([0, 0, 1, 1], [0.9, 0.9, 1, 1], action_steps)
    for i in range(action_steps):
        markers_thumb[i] = env.add_marker(
        {
        "pos": np.array([0.0, 0.0, 0.1]),
        "rgba": gradient_thumb[i],
        "type": 2, # shpere
        "size": [0.005, 0.005, 0.005]
        },
        )
        markers_index[i] = env.add_marker(
        {
        "pos": np.array([0.0, 0.0, 0.1]),
        "rgba": gradient_index[i],
        "type": 2, # shpere
        "size": [0.005, 0.005, 0.005]
        },
        )
        markers_middle[i] = env.add_marker(
        {
        "pos": np.array([0.0, 0.0, 0.1]),
        "rgba": gradient_middle[i],
        "type": 2, # shpere
        "size": [0.005, 0.005, 0.005]
        },
        )

    images = []
    with h5py.File(original_file, 'r') as hdf_file:
        sim_qpos = hdf_file['sim_state/qpos'][:]
        sim_qvel = hdf_file['sim_state/qvel'][:]
        sim_ctrl = hdf_file['sim_state/ctrl'][:]
        action = hdf_file['action'][:]
        env.set_goal(hdf_file["info/goal_position"][:], hdf_file["info/goal_orientation"][:])
        for i in range(len(sim_qpos)):
            env.data.qpos[:] = sim_qpos[i]
            env.data.qvel[:] = sim_qvel[i]
            env.data.ctrl[:] = sim_ctrl[i]
            for j in range(action_steps):
                try:
                    target_pos_thumb = get_pos_from_angle(action[i+j], env, 0)
                    markers_thumb[j].update("pos", target_pos_thumb)
                    target_pos_index = get_pos_from_angle(action[i+j], env, 120)
                    markers_index[j].update("pos", target_pos_index)
                    target_pos_middle = get_pos_from_angle(action[i+j], env, 240)
                    markers_middle[j].update("pos", target_pos_middle)
                except IndexError:
                    pass
            mujoco.mj_forward(env.model, env.data)
            images.append(env.render()[:, :, ::-1].astype(np.uint8))
    images = np.array(images, dtype=np.uint8)
    # Define frame properties
    height, width, _ = images.shape[1:]
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', width, height)
    for image in images:
        cv2.imshow('Video', image)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break
    cv2.destroyWindow('Video')
    output_file = 'test_dataset.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    for image in images:
        video_writer.write(image)
    video_writer.release()

def main(hdf5_path, task, fps):
    play_video_from_original(hdf5_path, task, fps)

if __name__ == "__main__":
    hdf5_path = "/home/local_rohan/OpenTeach/extracted_data/push_fix_low/demo_40/observation.h5"
    fps = 10
    task = "push_fix_low"
    main(hdf5_path, task, fps)