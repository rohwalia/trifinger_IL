import os
import h5py
import mujoco
import numpy as np
from trifinger_mujoco_env import MoveCubeEnv, TriFingerEnv
from matplotlib import pyplot as plt
from trifinger_mujoco_env.utils import get_keypoints_from_pose
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
    #env = TriFingerEnv(objects="no_objects", render_mode="rgb_array", dt=0.005, n_substeps=2, image_width=480, image_height=270, image_obs=True)
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
                        demo_num+=1
                        sim_qpos = existing_hdf['sim_state/qpos'][:]
                        sim_qvel = existing_hdf['sim_state/qvel'][:]
                        sim_ctrl = existing_hdf['sim_state/ctrl'][:]
                        print("Number of datapoints:", existing_hdf["info/num_datapoints"][()])
                        camera_images = np.full((len(sim_qpos), 3, 270, 360, 3), 0.0, dtype=np.uint8)
                        env.set_goal(existing_hdf["info/goal_position"][:], existing_hdf["info/goal_orientation"][:])
                        for i in range(len(sim_qpos)):
                            env.data.qpos[:] = sim_qpos[i]
                            env.data.qvel[:] = sim_qvel[i]
                            env.data.ctrl[:] = sim_ctrl[i]
                            mujoco.mj_forward(env.model, env.data)
                            camera_images[i] = env.get_camera_images()
                            # if i==1:
                            #     fig, axs = plt.subplots(1, 4)
                            #     for j, ax in enumerate(axs):
                            #         if j < 3:
                            #             ax.imshow(camera_images[i][j])
                            #     plt.show()
                        position = existing_hdf['robot/position'][:]
                        action = existing_hdf['action'][:]
                        keypoints = np.array([get_keypoints_from_pose(sim_qpos[i][9:12], quaternion.from_float_array(sim_qpos[i][12:16])).flatten() for i in range(len(sim_qpos))])
                        demo_group = output_hdf.create_group(f"demo_{demo_num}")

                        demo_group.create_dataset('camera_0', data=camera_images[:,0], compression="gzip", compression_opts = 6)
                        demo_group.create_dataset('camera_1', data=camera_images[:,1], compression="gzip", compression_opts = 6)
                        demo_group.create_dataset('camera_2', data=camera_images[:,2], compression="gzip", compression_opts = 6)
                        demo_group.create_dataset('robot_pose', data=position, compression="gzip", compression_opts = 6)
                        demo_group.create_dataset('cube_keypoints', data=keypoints, compression="gzip", compression_opts = 6)
                        demo_group.create_dataset('action', data=action, compression="gzip", compression_opts = 6)
    del env
    with open(os.path.splitext(output_hdf5_path)[0] + '.txt', 'w') as file:
        for item in filename_list:
            file.write(f"{item}\n")

# Specify the path to the directory containing demonstration data
input_dir = '/home/local_rohan/OpenTeach/extracted_data/push_fix_dim'
# Specify the name of the output HDF5 file
output_hdf5_path = '/home/local_rohan/consistency-policy/data/push_hybrid/push_hybrid.hdf5'

task = "push_fix"

# Call the function to merge HDF5 files
extract_and_merge_data(input_dir, output_hdf5_path, task)