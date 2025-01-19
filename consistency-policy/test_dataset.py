import h5py
import cv2
import numpy as np
import re
import argparse
import os
import matplotlib.pyplot as plt
from trifinger_mujoco_env import MoveCubeEnv, TriFingerEnv
import mujoco

def get_pos_from_angle(target_angle, env):
    kinematic_data = mujoco.MjData(env.model)
    kinematic_data.qpos[:9] = target_angle
    mujoco.mj_kinematics(env.model, kinematic_data)
    tip_positions = np.zeros((3, 3), dtype=np.float32)
    for i, alpha in enumerate([0, 120, 240]):
        tip_positions[i] = kinematic_data.geom(f"tip_{alpha}").xpos
    return tip_positions

def play_video_from_hdf5_group(group, camera_key, fps=20):
    # Check if the camera key exists in the group
    if camera_key not in group:
        print(f"Key '{camera_key}' not found in group '{group.name}'.")
        return
    
    # Load images from the specified camera key
    images = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in np.array(group[camera_key][:], dtype=np.uint8)])
    
    # Ensure images are numpy arrays
    if not isinstance(images, np.ndarray):
        print(f"Data under '{camera_key}' in group '{group.name}' is not a numpy array.")
        return

    # Define frame properties
    height, width, _ = images.shape[1:]

    # Initialize the video window
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', width, height)

    # Iterate over images and display them
    for image in images:
        cv2.imshow('Video', image)
        
        # Wait for the specified interval between frames
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    # Close the video window
    cv2.destroyWindow('Video')

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

def plot_relevant_val_from_original(original_file):
    with h5py.File(original_file, 'r') as hdf_file:
        reward_demo = hdf_file['rew'][:]
        has_achieved_demo = hdf_file['has_achieved'][:]
        plt.plot(reward_demo)
        plt.show()
        plt.plot(has_achieved_demo)
        plt.show()

def play_video_from_original(original_file, task, fps):
    if "_low" in task:
        task = task.replace("_low", "")
    if "_dim" in task:
        task = task.replace("_dim", "")
    if "_hybrid" in task:
        task = task.replace("_hybrid", "")
    if "_fix" in task:
        task = task.replace("_fix", "")
    env = MoveCubeEnv(task=task, render_mode="rgb_array", dt=0.005, n_substeps=2, image_width=480, image_height=270, image_obs=True)
    #env = TriFingerEnv(objects="no_objects", render_mode="rgb_array", dt=0.005, n_substeps=2, image_width=480, image_height=270, image_obs=True)
    marker_thumb = env.add_marker(
    {
        "pos": np.array([0.0, 0.0, 0.1]),
        "rgba": [1.0, 0.0, 0.0, 1.0],
        "type": 2, # shpere
        "size": [0.005, 0.005, 0.005]
    },
    )
    marker_index = env.add_marker(
        {
            "pos": np.array([0.0, 0.0, 0.1]),
            "rgba": [0.0, 1.0, 0.0, 1.0],
            "type": 2, # shpere
            "size": [0.005, 0.005, 0.005]
        },
    )
    marker_middle = env.add_marker(
        {
            "pos": np.array([0.0, 0.0, 0.1]),
            "rgba": [0.0, 0.0, 1.0, 1.0],
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
        plt.plot(sim_qpos[:,0], label="actual")
        plt.plot(action[:,0], label="desired")
        plt.legend()
        plt.show()
        env.set_goal(hdf_file["info/goal_position"][:], hdf_file["info/goal_orientation"][:])
        for i in range(len(sim_qpos)):
            env.data.qpos[:] = sim_qpos[i]
            env.data.qvel[:] = sim_qvel[i]
            env.data.ctrl[:] = sim_ctrl[i]
            target_pos = get_pos_from_angle(action[i], env)
            marker_thumb.update("pos", target_pos[0])
            marker_index.update("pos", target_pos[1])
            marker_middle.update("pos", target_pos[2])
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

def main(hdf5_path, camera_key, fps, demo_num):
    with open(os.path.splitext(hdf5_path)[0] + '.txt', 'r') as file:
        file_list = file.read().splitlines()
    # Open the HDF5 file
    with h5py.File(hdf5_path, 'r') as hdf_file:
        if demo_num is not None:
            print("Original file:", file_list[demo_num-1])
            group_name = f'demo_{demo_num}'
            if group_name in hdf_file:
                print(f"Playing video from group: {group_name}")
                play_video_from_hdf5_group(hdf_file[group_name], camera_key, fps)
                play_video_from_original(file_list[demo_num-1], os.path.splitext(os.path.basename(hdf5_path))[0], fps)
                plot_relevant_val_from_original(file_list[demo_num-1])
            else:
                print(f"Group '{group_name}' not found in the HDF5 file.")
        else:
            # Find the maximum demo number
            max_demo_num = find_max_demo_number(hdf_file)
            print(f"Maximum demo number: {max_demo_num}")
            
            # Iterate over all demo groups and play the corresponding videos
            for demo_num in range(1, max_demo_num + 1):
                group_name = f'demo_{demo_num}'
                if group_name in hdf_file:
                    print(f"Playing video from group: {group_name}")
                    play_video_from_hdf5_group(hdf_file[group_name], camera_key, fps)
                else:
                    print(f"Group '{group_name}' not found in the HDF5 file.")
            
            print("All videos have been played.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play videos from HDF5 file")
    parser.add_argument('hdf5_path', type=str, help='Path to HDF5 file')
    parser.add_argument('--camera_key', type=str, default='camera_0', help='Key for camera images')
    parser.add_argument('--fps', type=int, default=20, help='Frames per second for video playback')
    parser.add_argument('--demo_num', type=int, default=None, help='Specific demo number to play')

    args = parser.parse_args()
    main(args.hdf5_path, args.camera_key, args.fps, args.demo_num)

# python test_dataset.py /home/local_rohan/consistency-policy/data/push.hdf5 --demo_num 1 --camera_key camera_1 --fps 50