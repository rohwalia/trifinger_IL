import os
import h5py
import mujoco
import numpy as np
from trifinger_mujoco_env import MoveCubeEnv
from trifinger_mujoco_env.utils import get_keypoints_from_pose
from matplotlib import pyplot as plt
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
    if "_low" in task:
        task = task.replace("_low", "")
    if "_fix" in task:
        task = task.replace("_fix", "")
    env = MoveCubeEnv(task=task, render_mode=None, dt=0.005, n_substeps=2, image_width=480, image_height=270, image_obs=True)
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
                        #print("Frequency:", existing_hdf["info/record_frequency"][()])
                        demo_num+=1
                        sim_qpos = existing_hdf['sim_state/qpos'][:]
                        sim_ctrl = existing_hdf['sim_state/ctrl'][:]
                        keypoints = np.array([get_keypoints_from_pose(sim_qpos[i][9:12], quaternion.from_float_array(sim_qpos[i][12:16])).flatten() for i in range(len(sim_qpos))])
                        print(keypoints.shape)
                        print("Number of datapoints:", existing_hdf["info/num_datapoints"][()])
                        position = existing_hdf['robot/position'][:]
                        action = existing_hdf['action'][:]
                        demo_group = output_hdf.create_group(f"demo_{demo_num}")
                        demo_group.create_dataset('robot_pose', data=position, compression="gzip", compression_opts = 6)
                        demo_group.create_dataset('torques', data=sim_ctrl, compression="gzip", compression_opts = 6)
                        demo_group.create_dataset('cube_keypoints', data=keypoints, compression="gzip", compression_opts = 6)
                        demo_group.create_dataset('action', data=action, compression="gzip", compression_opts = 6)
    with open(os.path.splitext(output_hdf5_path)[0] + '.txt', 'w') as file:
        for item in filename_list:
            file.write(f"{item}\n")

# Specify the path to the directory containing demonstration data
input_dir = '/home/local_rohan/OpenTeach/extracted_data/push_fix'
# Specify the name of the output HDF5 file
output_hdf5_path = '/home/local_rohan/consistency-policy/data/push_low/push_low.hdf5'

task = "push_fix"

# Call the function to merge HDF5 files
extract_and_merge_data(input_dir, output_hdf5_path, task)