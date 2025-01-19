import h5py
import cv2
import numpy as np
import re
import argparse
import os
import matplotlib.pyplot as plt
from trifinger_mujoco_env import TriFingerEnv
import mujoco

def get_pos_from_angle(target_angle, env):
    kinematic_data = mujoco.MjData(env.model)
    kinematic_data.qpos[:9] = target_angle
    mujoco.mj_kinematics(env.model, kinematic_data)
    tip_positions = np.zeros((3, 3), dtype=np.float32)
    for i, alpha in enumerate([0, 120, 240]):
        tip_positions[i] = kinematic_data.geom(f"tip_{alpha}").xpos
    return tip_positions

def play_video_from_original(original_file, fps):

    env = TriFingerEnv(objects="no_objects", render_mode="rgb_array", dt=0.005, n_substeps=2, image_width=480, image_height=270, image_obs=True)
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
        for i in range(len(sim_qpos[0])):
            if i==0:
                plt.plot(sim_qpos[:,i], label="actual")
                plt.plot(action[:,i], label="desired")
                plt.legend()
                plt.show()
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

def main(hdf5_path, fps):
    play_video_from_original(hdf5_path, fps)

if __name__ == "__main__":
    hdf5_path = "/home/local_rohan/OpenTeach/extracted_data/test/demo_0/observation.h5"
    fps = 10
    main(hdf5_path, fps)