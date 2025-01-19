import os
import h5py
import mujoco
import numpy as np
from trifinger_mujoco_env import MoveCubeEnv, TriFingerEnv
from matplotlib import pyplot as plt
from trifinger_mujoco_env.utils import get_keypoints_from_pose
import cv2
import quaternion

def extract_and_merge_data(input_dir, output_hdf5_path, task):
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_hdf5_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Remove the existing output HDF5 file if it exists
    if os.path.exists(output_hdf5_path):
        os.remove(output_hdf5_path)

    demo_num=0
    filename_list = []

    # Create or open the output HDF5 file
    with h5py.File(output_hdf5_path, 'w') as output_hdf:
        # Traverse the directory structure
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.hdf5') or file.endswith('.h5'):
                    # demo_id = os.path.basename(root)
                    file_path = os.path.join(root, file)
                    print(file_path)
                    filename_list.append(file_path)
                    # Open the existing HDF5 file in read-only mode
                    with h5py.File(file_path, 'r') as existing_hdf:
                        # Check if the keys exist
                        if 'camera_observation' in existing_hdf.keys() and 'robot_observation/position' in existing_hdf.keys():
                            # Read the camera images and robot positions
                            demo_num+=1
                            camera_images = existing_hdf['camera_observation'][:]
                            position = existing_hdf['robot_observation/position'][:]
                            torque = existing_hdf['robot_observation/torque'][:]
                            action = existing_hdf['action'][:]

                            # Ensure there are at least 3 images
                            if camera_images.shape[0] < 3:
                                raise ValueError(f"Not enough images in 'camera_images' in {file_path} to separate as camera_0, camera_1, and camera_2")

                            # Create a group for the demonstration_id
                            demo_group = output_hdf.create_group(f"demo_{demo_num}")

                            # Save images separately
                            demo_group.create_dataset('camera_0', data=camera_images[:,0], compression="gzip", compression_opts = 6)
                            demo_group.create_dataset('camera_1', data=camera_images[:,1], compression="gzip", compression_opts = 6)
                            demo_group.create_dataset('camera_2', data=camera_images[:,2], compression="gzip", compression_opts = 6)
                            demo_group.create_dataset('robot_pose', data=position, compression="gzip", compression_opts = 6)
                            demo_group.create_dataset('torques', data=torque, compression="gzip", compression_opts = 6)
                            demo_group.create_dataset('action', data=action, compression="gzip", compression_opts = 6)
                        else:
                            raise KeyError(f"'camera_images'/'sim_state' or 'robot/position' keys not found in {file_path}.")
    with open(os.path.splitext(output_hdf5_path)[0] + '.txt', 'w') as file:
        for item in filename_list:
            file.write(f"{item}\n")

# Specify the path to the directory containing demonstration data
input_dir = '/home/local_rohan/OpenTeach/extracted_data/real_push'
# Specify the name of the output HDF5 file
output_hdf5_path = '/home/local_rohan/consistency-policy/data/real_push/real_push.hdf5'

task = "push"

# Call the function to merge HDF5 files
extract_and_merge_data(input_dir, output_hdf5_path, task)