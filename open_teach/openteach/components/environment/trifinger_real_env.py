import os
import re
import time
import numpy as np
from openteach.utils.timer import FrequencyTimer
from openteach.utils.network import ZMQCameraPublisher, ZMQCompressedImageTransmitter,ZMQKeypointPublisher,ZMQKeypointSubscriber
from openteach.components.environment.arm_env import Arm_Env
from openteach.constants import *
from openteach.utils.images import rescale_image
from pathlib import Path

import h5py
import copy
import matplotlib.pyplot as plt

import socket
import typing
import logging
from ament_index_python.packages import get_package_share_directory
import robot_fingers
import robot_properties_fingers
from trifinger_rl_datasets import TriFingerDatasetEnv, PolicyBase
from trifinger_rl_datasets.sim_env import SimTriFingerCubeEnv

try:
    clip = np.core.umath.clip  # type: ignore
except AttributeError:
    clip = np.clip

class TriFingerFrontendWithKinematics:
	"""Frontend which also has a kinematics attribute.

	Used in RealTriFingerCubeEnv"""

	def __init__(
		self, kinematics: robot_properties_fingers.pinocchio_utils.Kinematics
	) -> None:
		self._platform = robot_fingers.TriFingerPlatformFrontend() #TriFingerPlatformWithObjectFrontend()
		self.append_desired_action = self._platform.append_desired_action
		self.wait_until_timeindex = self._platform.wait_until_timeindex
		self.get_robot_observation = self._platform.get_robot_observation
		self.get_camera_observation = self._platform.get_camera_observation
		self.get_current_timeindex = self._platform.get_current_timeindex
		self.get_timestamp_ms = self._platform.get_timestamp_ms
		self.Action = self._platform.Action

		self.kinematics = kinematics
		self.forward_kinematics = self.kinematics.forward_kinematics


class RealTriFingerCubeEnv(SimTriFingerCubeEnv):
	"""
	Gym environment for manipulation of a cube with a real TriFingerPro platform.

	Derives from SimTriFingerCubeEnv and overwrites some methods which are
	different on the real robot.
	"""

	def __init__(self, **kwargs) -> None:
		super().__init__(**kwargs)
		# overwrite robot id
		self.robot_id = self._get_robot_id()
		# kinematics
		self.kinematics = self._get_kinematics()

	def _get_robot_id(self) -> int:
		"""Get ID of robot the environment is running on."""
		# get robot ID from hostname
		hostname = socket.gethostname()
		m = re.match(r"^(?:roboch)(\d+)$", hostname)
		if m:
			robot_id = int(m.group(1))
		else:
			# if not on real robots use robot id 0
			robot_id = 0

		logging.info("Robot ID: %d", robot_id)

		return robot_id

	def _get_kinematics(
		self,
	) -> robot_properties_fingers.pinocchio_utils.Kinematics:
		"""Get kinematics object."""
		tip_link_names = [
			"finger_tip_link_0",
			"finger_tip_link_120",
			"finger_tip_link_240",
		]

		robot_properties_path = "/opt/conda/envs/openteach/lib/python3.10/site-packages/robot_properties_fingers" #get_package_share_directory("robot_properties_fingers")
		finger_urdf_path = os.path.join(
			robot_properties_path, "urdf/pro/trifingerpro.urdf"
		)

		return robot_properties_fingers.pinocchio_utils.Kinematics(
			finger_urdf_path, tip_link_names
		)

	def _append_desired_action(self, robot_action: typing.Sequence[float]) -> int:
		"""Append desired action to queue and wait for timeindex.

		The waiting is required to avoid an error about accessing
		time_series elements which are too old.
		"""
		assert self.platform is not None

		t = self.platform.append_desired_action(robot_action)
		self._wait_until_timeindex(t)
		return t

	def _wait_until_timeindex(self, t: int) -> None:
		self.platform.wait_until_timeindex(t)  # type: ignore

	def reset(  # type: ignore
		self
	) -> typing.Union[dict, typing.Tuple[dict, dict]]:
		# cannot reset multiple times
		if self.platform is not None:
			raise RuntimeError("Once started, this environment cannot be reset.")
		self.platform = TriFingerFrontendWithKinematics(self.kinematics)  # type: ignore
		self.step_count = 0
		# TODO: self._step_size = (1/frequency)*1000

		# need to already do one step to get initial observation
		self.t_obs = 0
		obs, info = self.step(self._initial_action)
		info = {"time_index": -1}

		return obs, info

	def _get_pose_delay(self, camera_observation, t: int) -> float:
		"""Get delay between when the object pose was caputered and now."""

		# real robot uses time since epoch as timestamp for camera images
		return time() - camera_observation.cameras[0].timestamp

	def _wait_until_timeindex(self, t: int):
		"""Wait until the given time index is reached."""
		self.platform.wait_until_timeindex(t)  # type: ignore


class RealTriFingerDatasetEnv(TriFingerDatasetEnv):
	"""TriFingerDataset environment with real instead of sim environment."""

	def __init__(self, **kwargs) -> None:
		super().__init__(**kwargs)

		# replace sim env with real env (have to use self.t_kwargs to make sure
		# that image obs are used when requested).
		self.sim_env = RealTriFingerCubeEnv(**self.t_kwargs)

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

class TriFingerRealEnv(Arm_Env):
	def __init__(self,
			 host,
			 camport,
			 task,
			 endeffpossubscribeport,
			 stream_oculus,
			 collect,
			 storage_path,
			 user,
	):
		self._timer=FrequencyTimer(VR_FREQ)
		self.host=host
		self.camport=camport
		self.task = task
		self.stream_oculus=stream_oculus

		self._stream_oculus = stream_oculus
		self.target = np.array([0,1,2])

		self.frequency = 10 #50
		self.last_time = time.time()
		self.run = False
		self.collect = collect
		self.user = user
		if self.collect:
			self.robot_information = dict()
			self.storage_path = storage_path
		#Define ZMQ pub/sub
		#Publishing the stream into the oculus.
		if self._stream_oculus:
			self.rgb_viz_publisher = ZMQCompressedImageTransmitter(
				host = host,
				port = camport + VIZ_PORT_OFFSET
			)

		self.endeff_pos_subscriber = ZMQKeypointSubscriber(
			host = host,
			port = endeffpossubscribeport,
			topic='endeff_coords'
		)

		self.reset_subscriber = ZMQKeypointSubscriber(
			host = host,
			port = endeffpossubscribeport,
			topic='reset'
		)

		self.start_subscriber = ZMQKeypointSubscriber(
			host = host,
			port = endeffpossubscribeport,
			topic='start'
		)
		
		self.stop_subscriber = ZMQKeypointSubscriber(
			host = host,
			port = endeffpossubscribeport,
			topic='stop'
		)

		self.name="Trifinger"
		# initialize env
		env_args = {
			"name": "trifinger-teleop",
			"dataset_url": None,
			"ref_min_score": 0.0,
			"ref_max_score": 1.0 * 30000 / 20,
			"real_robot": False,  # To keep stepping enabled
			"trifinger_kwargs": {
			    # "episode_length": 1500,
			    # "difficulty": 4,
			    "keypoint_obs": False,
			    "obs_action_delay": 10,
			},
			"flatten_obs": False,
			"image_obs": True,
		}
		self.env = RealTriFingerDatasetEnv(**env_args)
		print("Resetting Robot")
		_, _ = self.env.reset()
		self.obs, _ = self.env.reset_fingers(reset_wait_time=500)
					
	# Reset the environment
	def reset(self):
		obs, _ = self.env.reset_fingers(reset_wait_time=1000)
		return obs
	
	# Get the time
	def get_time(self):
		return time.time()
	
			
	@property              
	def timer(self):
		return self._timer
	
	def check_target_pos(self, target_pos, margin):
		r = 0.195
		h = 0.027
		h_l = 0.148 #0.175-h
		r_l = 0.278

		def move_to_boundary(x, y, radius, margin):
			distance = np.sqrt(x**2 + y**2)
			adjusted_radius = radius - margin
			if distance > adjusted_radius:
				scale_factor = adjusted_radius / distance
				return x * scale_factor, y * scale_factor
			return x, y

		target_valid = np.copy(target_pos)

		for pos in target_valid:
			x, y, z = pos

			if z < 0:
				pos[2] = margin
				pos[0], pos[1] = move_to_boundary(x, y, r, margin)
			elif z <= h:
				if x**2 + y**2 > r**2:
					pos[0], pos[1] = move_to_boundary(x, y, r, margin)
			elif z <= h + h_l:
				radius_at_z = r + (r_l - r) * (z - h) / h_l
				if x**2 + y**2 > radius_at_z**2:
						pos[0], pos[1] = move_to_boundary(x, y, radius_at_z, margin)

		return target_valid
	
	# Take action
	def take_action(self):
		target_pos = np.array(self.endeff_pos_subscriber.recv_keypoints())
		reset = self.reset_subscriber.recv_keypoints()
		if reset:
			self.target[0] = np.argmax(target_pos[:,1])
			self.target[2] = np.argmin(target_pos[:,0])
			self.target[1] = 3-self.target[0]-self.target[2] #np.argmax(target_pos[:,0])
			print(self.target)
		target_pos_retarget = target_pos[self.target]
		target_valid = self.check_target_pos(target_pos_retarget, margin=0.015)
		target_angle, _ = self.env.sim_env.kinematics.inverse_kinematics(target_valid, self.obs["robot_observation"]["position"])#.reshape(9)
		action = np.asarray(target_angle, dtype=np.float32)
		action = clip(action, self.env.action_space.low, self.env.action_space.high)
		obs, _ = self.env.step(action)
		obs["action"] = target_angle
		return obs
	
	def get_recorder_file_name(self):
		prefix = Path(__file__).resolve().parents[3] / self.storage_path / self.task
		items = os.listdir(prefix)
		pattern = re.compile(r'^demo_(\d+)$')
		max_number = -1
	
		for item in items:
			match = pattern.match(item)
			if match:
				number = int(match.group(1))
				if number > max_number:
					max_number = number

		storage_path = self._storage_path = os.path.join(
			prefix,
			'demo_{}'.format(max_number+1)
		)
		if os.path.exists(storage_path):
			pass
		else:
			os.makedirs(storage_path)
		recorder_file_name = os.path.join(storage_path, "observation" + '.h5')
		return recorder_file_name
	
	def save_data(self):
		print('Compressing keypoint data...')
		self.robot_information["info/num_datapoints"] = len(self.robot_information['action'])
		self.robot_information["info/record_duration"] = self.robot_information['timestamp'][-1] - self.robot_information['timestamp'][0]
		self.robot_information["info/record_frequency"] = len(self.robot_information['action']) / (self.robot_information['timestamp'][-1] - self.robot_information['timestamp'][0])
		print(self.robot_information["info/record_frequency"])
		# if self.task != "none":
		# 	self.robot_information["info/goal_position"] = self.env.goal_position
		# 	self.robot_information["info/goal_orientation"] = self.env.goal_orientation
		self.robot_information["info/user"] = self.user
		recorder_file_name = self.get_recorder_file_name()
		with h5py.File(recorder_file_name, "w") as file:
			for key in self.robot_information.keys():
				if key in ["info/num_datapoints", "info/record_duration", "info/record_frequency", "info/user"]:
					if key=="info/user":
						file.create_dataset(key, data = self.robot_information[key], dtype=h5py.string_dtype(encoding='utf-8'))
					else:
						file.create_dataset(key, data = self.robot_information[key])
				elif key != 'timestamp':
					self.robot_information[key] = np.array(self.robot_information[key], dtype = np.float32)
					file.create_dataset(key, data = self.robot_information[key], compression="gzip", compression_opts = 6)
				else:
					self.robot_information['timestamp'] = np.array(self.robot_information['timestamp'], dtype = np.float64)
					file.create_dataset('timestamp', data = self.robot_information['timestamp'], compression="gzip", compression_opts = 6)
			print('Saved keypoint data in {}.'.format(recorder_file_name))

	# Stream the environment
	def stream(self):
		self.notify_component_start('{} environment'.format(self.name))
		while True:
			try:
				self.timer.start_loop()
				delay = (1/self.frequency)-(time.time()-self.last_time)
				if delay>0:
					time.sleep(delay)
				self.last_time = time.time()
				self.obs = self.take_action()
				self.obs["timestamp"] = self.get_time()
				if self._stream_oculus:
					self.rgb_viz_publisher.send_image(self.obs["camera_observation"][1][:, :, ::-1])
				if self.collect:
					start = self.start_subscriber.recv_keypoints()
					stop = self.stop_subscriber.recv_keypoints()
					if (start == True and self.run == False) or (self.run==True and stop==False):
						if self.run == False:
							print("Starting recording...")
							self.run = True
						if not self.robot_information:
							self.keys = get_nested_keys(self.obs)
						for key in self.keys:
							if key not in self.robot_information.keys():
								self.robot_information[key] = [copy.deepcopy(get_value_from_nested_dict(self.obs, key))]
							else:
								self.robot_information[key].append(copy.deepcopy(get_value_from_nested_dict(self.obs, key)))
					if stop == True and self.run == True:
						print("Stopping recording...")
						self.run = False
						self.save_data()
						self.robot_information = dict()
						self.reset()

				self.timer.end_loop()
			except KeyboardInterrupt:
					break
		# Writing to dataset
		if self.collect:
			self.save_data()
		if self._stream_oculus:
			self.rgb_viz_publisher.stop()
		print('Stopping the environment!')
