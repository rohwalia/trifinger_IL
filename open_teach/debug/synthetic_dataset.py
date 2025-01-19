from trifinger_mujoco_env import TriFingerEnv as MujocoEnv
import h5py
import copy
import matplotlib.pyplot as plt
import numpy as np
import mujoco

def get_pos_from_angle(target_angle, env):
    kinematic_data = mujoco.MjData(env.model)
    kinematic_data.qpos[:9] = target_angle
    mujoco.mj_kinematics(env.model, kinematic_data)
    tip_positions = np.zeros((3, 3), dtype=np.float32)
    for i, alpha in enumerate([0, 120, 240]):
        tip_positions[i] = kinematic_data.geom(f"tip_{alpha}").xpos
    return tip_positions

def get_nested_keys(d, parent_key='', separator='/'):
    keys = []
    for k, v in d.items():
        current_key = f"{parent_key}{separator}{k}" if parent_key else k
        if isinstance(v, dict):
            keys.extend(get_nested_keys(v, current_key, separator))
        else:
            keys.append(current_key)
    return keys

def get_value_from_nested_dict(d, key_string, separator='/'):
    keys = key_string.split(separator)
    current = d
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return None

def take_action(env, target_pos):
    frequency=10
    dt=0.005
    p_gain = np.array([2.5, 2.5, 1.5] * 3)
    d_gain = np.array([0.1, 0.2, 0.1] * 3)
    obs = env.get_observation(detail = False, state=True)
    target_angle = env.get_angles_from_tip_pos(target_pos).reshape(9)
    obs["action"] = target_angle
    obs_temp = obs
    for i in range(int(1/(frequency*dt*2))):
        delta = target_angle - obs_temp["robot"]["position"]
        action = p_gain * delta - d_gain * obs_temp["robot"]["velocity"]  #(delta-self.prev_delta)/0.02
        if i==int(1/(frequency*dt*2))-1:
            _, obs["rew"], _, _, obs["has_achieved"] = env.step(action, return_obs=True)
        else:
            _, _, _, _, _ = env.step(action, return_obs=False)
        obs_temp = env.get_observation(detail = False, state=True)
    return obs


env = MujocoEnv(objects="no_objects", render_mode="human", dt=0.005, n_substeps=2, image_width=480, image_height=270, image_obs=True) # if changed also change value of IMAGE_RECORD_RESOLUTION_SIM in constants.py
obs, _ = env.reset()
initial = get_pos_from_angle(obs["robot"]["position"], env)
print(initial)
goal = np.array([[0.14, 0.0, 0.065/2], [0.0, 0.14, 0.065/2], [-0.14, 0.0, 0.065/2]])
goal_duration = 20
path_duration = 15
robot_information = dict()
episode_length = goal_duration+path_duration
path = np.linspace(initial, goal, path_duration)
actions = np.vstack([path, [goal] * goal_duration])


for i in range(episode_length):
    obs = take_action(env, actions[i])
    if not robot_information:
        keys = get_nested_keys(obs)
    for key in keys:
        if key not in robot_information.keys():
            robot_information[key] = [copy.deepcopy(get_value_from_nested_dict(obs, key))]
        else:
            robot_information[key].append(copy.deepcopy(get_value_from_nested_dict(obs, key)))


robot_information["info/num_datapoints"] = len(robot_information['action'])
recorder_file_name = "/home/local_rohan/OpenTeach/extracted_data/test/demo_0/observation.h5"
with h5py.File(recorder_file_name, "w") as file:
    for key in robot_information.keys():
        if key in ["info/num_datapoints", "info/record_duration", "info/record_frequency", "info/user"]:
            if key=="info/user":
                file.create_dataset(key, data = robot_information[key], dtype=h5py.string_dtype(encoding='utf-8'))
            else:
                file.create_dataset(key, data = robot_information[key])
        elif key != 'timestamp':
            robot_information[key] = np.array(robot_information[key], dtype = np.float32)
            file.create_dataset(key, data = robot_information[key], compression="gzip", compression_opts = 6)
        else:
            robot_information['timestamp'] = np.array(robot_information['timestamp'], dtype = np.float64)
            file.create_dataset('timestamp', data = robot_information['timestamp'], compression="gzip", compression_opts = 6)