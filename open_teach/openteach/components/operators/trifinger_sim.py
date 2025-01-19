import numpy as np
import matplotlib.pyplot as plt
import zmq

from tqdm import tqdm

from copy import deepcopy as copy
from openteach.constants import *
from openteach.utils.timer import FrequencyTimer
from openteach.utils.network import ZMQKeypointSubscriber, ZMQKeypointPublisher
from openteach.utils.vectorops import *
from openteach.utils.files import *
from scipy.spatial.transform import Rotation, Slerp
from .operator import Operator

np.set_printoptions(precision=2, suppress=True)


# Rotation should be filtered when it's being sent
class Filter:
	def __init__(self, state, comp_ratio=0.6):
		self.pos_state = state[:3]
		self.ori_state = state[3:7]
		self.comp_ratio = comp_ratio

	def __call__(self, next_state):
		self.pos_state = self.pos_state[:3] * self.comp_ratio + next_state[:3] * (1 - self.comp_ratio)
		ori_interp = Slerp([0, 1], Rotation.from_quat(
			np.stack([self.ori_state, next_state[3:7]], axis=0)),)
		self.ori_state = ori_interp([1 - self.comp_ratio])[0].as_quat()
		return np.concatenate([self.pos_state, self.ori_state])
	
class TrifingerSimOperator(Operator):
	def __init__(
		self,
		host,
		transformed_keypoints_port,
		transformed_keypoints_port_left,
		stream_configs,
		finger_right,
		finger_left,
		stream_oculus,
		endeff_publish_port,
		scale,
	):
		self.notify_component_start('Trifinger operator')
		self.finger_right = finger_right
		self.finger_left = finger_left
		self._host, self._port, self._port_left = host, transformed_keypoints_port, transformed_keypoints_port_left
		self._hand_transformed_keypoint_subscriber = ZMQKeypointSubscriber(
			host = self._host,
			port = self._port,
			topic = 'transformed_hand_coords'
		)
		self._hand_keypoint_subscriber = ZMQKeypointSubscriber(
			host = self._host,
			port = self._port,
			topic = 'hand_coords'
		)
		self._hand_keypoint_subscriber_left = ZMQKeypointSubscriber(
			host = self._host,
			port = self._port_left,
			topic = 'hand_coords'
		)
		self._hand_transformed_keypoint_subscriber_left = ZMQKeypointSubscriber(
			host = self._host,
			port = self._port_left,
			topic = 'transformed_hand_coords'
		)


		# Initalizing the robot controller
		self.resolution_scale = 1 # NOTE: Get this from a socket
		self.arm_teleop_state = ARM_TELEOP_STOP # We will start as the cont
		self.pause_flag=0
		self.prev_pause_flag=0
		self.pause_cnt=0
		self.scale = scale

		self.end_eff_position_publisher = ZMQKeypointPublisher(
			host = host,
			port = endeff_publish_port
		)

		self._stream_oculus=stream_oculus
		self.stream_configs=stream_configs
		self._timer = FrequencyTimer(VR_FREQ)
		self._robot='Trifinger_Sim'
		self.is_first_frame = True
		# self.detail_time = []
		
		# Frequency timer
		self._timer = FrequencyTimer(VR_FREQ)
		self.direction_counter = 0
		self.current_direction = 0

	@property
	def timer(self):
		return self._timer

	@property
	def robot(self):
		return self._robot
	
	@property
	def hand_keypoint_subscriber_left(self):
		return self._hand_keypoint_subscriber_left
	@property
	def transformed_hand_keypoint_subscriber(self):
		return self._hand_transformed_keypoint_subscriber
	@property
	def transformed_hand_keypoint_subscriber_left(self):
		return self._hand_transformed_keypoint_subscriber_left
	@property
	def hand_keypoint_subscriber(self):
		return self._hand_keypoint_subscriber
	
	def _get_hand_coords(self):
		for i in range(10):
			data = self.hand_keypoint_subscriber.recv_keypoints(flags=zmq.NOBLOCK)
			if not data is None: break 
		if data is None: return None
		return np.asanyarray(data)
	
	def _get_hand_coords_left(self):
		for i in range(10):
			data = self.hand_keypoint_subscriber_left.recv_keypoints(flags=zmq.NOBLOCK)
			if not data is None: break 
		if data is None: return None
		return np.asanyarray(data)
	 
	# Check if it is real robot or simulation
	def return_real(self):
		return False
			
	# Reset Teleoperation
	def _reset_teleop(self):
		# Just updates the beginning position of the arm
		print('****** RESETTING TELEOP ****** ')
		self.robot_init_coords= np.array([0,0.1,-0.1])
		first_hand_coords, first_hand_coords_left = self._get_hand_coords(), self._get_hand_coords_left()
		while first_hand_coords is None or first_hand_coords_left is None:
			first_hand_coords, first_hand_coords_left = self._get_hand_coords(), self._get_hand_coords_left()

		self.rot= np.array([[1,0,0], 
					  		[0,0,1], 
							[0,1,0]])
		self.trans = self.robot_init_coords-first_hand_coords[OCULUS_JOINTS['thumb'][-1]]
		self.is_first_frame = False

		return first_hand_coords, first_hand_coords_left
	
	# Get ARm Teleop state from Hand keypoints 
	def _get_arm_teleop_state_from_hand_keypoints(self):
		pause_state ,pause_status,pause_right =self.get_pause_state_from_hand_keypoints()
		pause_status =np.asanyarray(pause_status).reshape(1)[0] 
		return pause_state,pause_status,pause_right
	
	# Get Pause State from Hand Keypoints 
	def get_pause_state_from_hand_keypoints(self):
		transformed_hand_coords= self.transformed_hand_keypoint_subscriber.recv_keypoints()
		middle_distance = np.linalg.norm(transformed_hand_coords[OCULUS_JOINTS['middle'][-1]]- transformed_hand_coords[OCULUS_JOINTS['thumb'][-1]])
		thresh = 0.02 
		pause_right= True
		if middle_distance < thresh:
			self.pause_cnt+=1
			if self.pause_cnt==1:
				self.prev_pause_flag=self.pause_flag
				self.pause_flag = not self.pause_flag       
		else:
			self.pause_cnt=0
		pause_state = np.asanyarray(self.pause_flag).reshape(1)[0]
		pause_status= False  
		if pause_state!= self.prev_pause_flag:
			pause_status= True 
		return pause_state , pause_status , pause_right
	
	# Apply retargeted angles
	def _apply_retargeted_angles(self, log=False):
		# See if there is a reset in the teleop
		new_arm_teleop_state,pause_status,pause_right = self._get_arm_teleop_state_from_hand_keypoints()
		if self.is_first_frame or (self.arm_teleop_state == ARM_TELEOP_STOP and new_arm_teleop_state == ARM_TELEOP_CONT):
			hand_coords, hand_coords_left = self._reset_teleop() # Should get the moving hand frame only once
			reset = True
		else:
			hand_coords, hand_coords_left = self._get_hand_coords(), self._get_hand_coords_left()
			reset = False
		self.arm_teleop_state = new_arm_teleop_state

		if hand_coords is None or hand_coords_left is None: 
			return # It means we are not on the arm mode yet instead of blocking it is directly returning
		
		target_coords = []
		for i in self.finger_right:
			target_coords.append(((self.rot@(hand_coords[OCULUS_JOINTS[i][-1]]+self.trans).T).T) * np.array([self.scale, self.scale, 1]))

		for i in self.finger_left:
			target_coords.append(((self.rot@(hand_coords_left[OCULUS_JOINTS[i][-1]]+self.trans).T).T) * np.array([self.scale, self.scale, 1]))
		
		transformed_hand_coords_left = self.transformed_hand_keypoint_subscriber_left.recv_keypoints()
		if np.linalg.norm(transformed_hand_coords_left[OCULUS_JOINTS['index'][-1]]- transformed_hand_coords_left[OCULUS_JOINTS['thumb'][-1]]) < 0.02:
			start = True
		else:
			start = False
		if np.linalg.norm(transformed_hand_coords_left[OCULUS_JOINTS['middle'][-1]]- transformed_hand_coords_left[OCULUS_JOINTS['thumb'][-1]]) < 0.02:
			stop = True
		else:
			stop = False

		if np.linalg.norm(transformed_hand_coords_left[OCULUS_JOINTS['pinky'][-1]]- transformed_hand_coords_left[OCULUS_JOINTS['thumb'][-1]]) < 0.02:
			raise KeyboardInterrupt

		if self.arm_teleop_state == ARM_TELEOP_CONT:
			self.end_eff_position_publisher.pub_keypoints(target_coords,"endeff_coords")
			self.end_eff_position_publisher.pub_keypoints(reset,"reset")
			self.end_eff_position_publisher.pub_keypoints(start,"start")
			self.end_eff_position_publisher.pub_keypoints(stop,"stop")