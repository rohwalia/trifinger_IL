import h5py
import cv2
import numpy as np
import re
import argparse
import os
import matplotlib.pyplot as plt
from trifinger_mujoco_env import MoveCubeEnv, TriFingerEnv
import mujoco

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
    print(height)
    print(width)

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

                print(hdf_file[group_name]["robot_pose"].shape)
                print(hdf_file[group_name]["action"].shape)
                print(hdf_file[group_name]["torques"].shape)

                plt.plot(hdf_file[group_name]["robot_pose"][:,0], label="actual")
                plt.plot(hdf_file[group_name]["action"][:,0], label="desired")
                plt.legend()
                plt.show()
            else:
                print(f"Group '{group_name}' not found in the HDF5 file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play videos from HDF5 file")
    parser.add_argument('hdf5_path', type=str, help='Path to HDF5 file')
    parser.add_argument('--camera_key', type=str, default='camera_0', help='Key for camera images')
    parser.add_argument('--fps', type=int, default=20, help='Frames per second for video playback')
    parser.add_argument('--demo_num', type=int, default=None, help='Specific demo number to play')

    args = parser.parse_args()
    main(args.hdf5_path, args.camera_key, args.fps, args.demo_num)

# python test_dataset.py /home/local_rohan/consistency-policy/data/push.hdf5 --demo_num 1 --camera_key camera_1 --fps 50