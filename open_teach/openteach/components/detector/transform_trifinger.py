import numpy as np
from copy import deepcopy as copy
from openteach.components import Component
from openteach.constants import *
from openteach.utils.vectorops import *
from openteach.utils.network import ZMQKeypointPublisher, ZMQKeypointSubscriber,ZMQButtonFeedbackSubscriber
from openteach.utils.timer import FrequencyTimer
import time
import matplotlib.pyplot as plt

class TransformHandPositionCoords(Component):
    def __init__(self, host, keypoint_port, transformation_port, transformation_port_left, moving_average_limit = 5):
        self.notify_component_start('keypoint position transform')
        
        # Initializing the subscriber for right hand keypoints
        self.original_keypoint_subscriber = ZMQKeypointSubscriber(host, keypoint_port, 'right')
        self.original_keypoint_subscriber_left = ZMQKeypointSubscriber(host, keypoint_port, 'left')
        # Initializing the publisher for transformed right hand keypoints
        self.transformed_keypoint_publisher = ZMQKeypointPublisher(host, transformation_port)
        self.transformed_keypoint_publisher_left = ZMQKeypointPublisher(host, transformation_port_left)
        # Timer
        self.timer = FrequencyTimer(VR_FREQ)
        # Keypoint indices for knuckles
        self.knuckle_points = (OCULUS_JOINTS['knuckles'][0], OCULUS_JOINTS['knuckles'][-1])
        # Moving average queue
        self.moving_average_limit = moving_average_limit
        # Create a queue for moving average
        self.coord_moving_average_queue = []
        self.coord_left_moving_average_queue = []
        self.coord_original_moving_average_queue = []
        self.coord_original_left_moving_average_queue = []
        # self.detail_time = []

    # Function to get the hand coordinates from the VR
    def _get_hand_coords(self, hand):
        if hand == "right":
            data = self.original_keypoint_subscriber.recv_keypoints()
        elif hand == "left":
            data = self.original_keypoint_subscriber_left.recv_keypoints()
        else:
            return None
        return np.asanyarray(data[1:]).reshape(OCULUS_NUM_KEYPOINTS, 3)
    
    # Function to find hand coordinates with respect to the wrist
    def _translate_coords(self, hand_coords):
        return copy(hand_coords) - hand_coords[0]

    # Create a coordinate frame for the hand
    def _get_coord_frame(self, index_knuckle_coord, pinky_knuckle_coord):
        palm_normal = normalize_vector(np.cross(index_knuckle_coord, pinky_knuckle_coord))   # Current Z
        palm_direction = normalize_vector(index_knuckle_coord + pinky_knuckle_coord)         # Current Y
        cross_product = normalize_vector(np.cross(palm_direction, palm_normal))              # Current X
        return [cross_product, palm_direction, palm_normal]
    
    def transform_keypoints(self, hand_coords):
        translated_coords = self._translate_coords(hand_coords)
        original_coord_frame = self._get_coord_frame(
            translated_coords[self.knuckle_points[0]], 
            translated_coords[self.knuckle_points[1]]
        )

        # Finding the rotation matrix and rotating the coordinates
        rotation_matrix = np.linalg.solve(original_coord_frame, np.eye(3)).T
        transformed_hand_coords = (rotation_matrix @ translated_coords.T).T

        return transformed_hand_coords

    def stream(self):
        while True:
            try:
                self.timer.start_loop()
                # start = time.time()
                hand_coords = self._get_hand_coords(hand="right")
                hand_coords_left = self._get_hand_coords(hand="left")
               
                # Shift the points to required axes
                transformed_hand_coords= self.transform_keypoints(hand_coords)
                transformed_hand_coords_left = self.transform_keypoints(hand_coords_left)

                # Passing the transformed coords into a moving average
                self.averaged_hand_coords = moving_average(
                    transformed_hand_coords, 
                    self.coord_moving_average_queue, 
                    self.moving_average_limit
                )

                self.averaged_hand_coords_left = moving_average(
                    transformed_hand_coords_left, 
                    self.coord_left_moving_average_queue, 
                    self.moving_average_limit
                )

                self.averaged_hand_coords_original = moving_average(
                    hand_coords, 
                    self.coord_original_moving_average_queue, 
                    self.moving_average_limit
                )

                self.averaged_hand_coords_original_left = moving_average(
                    hand_coords_left, 
                    self.coord_original_left_moving_average_queue, 
                    self.moving_average_limit
                )

                self.transformed_keypoint_publisher.pub_keypoints(self.averaged_hand_coords, 'transformed_hand_coords')
                self.transformed_keypoint_publisher_left.pub_keypoints(self.averaged_hand_coords_left, 'transformed_hand_coords')
                self.transformed_keypoint_publisher.pub_keypoints(self.averaged_hand_coords_original, 'hand_coords')
                self.transformed_keypoint_publisher_left.pub_keypoints(self.averaged_hand_coords_original_left, 'hand_coords')
                # self.detail_time.append(time.time()-start)
                self.timer.end_loop()
            except:
                break
        # plt.boxplot(self.detail_time)
        # plt.show()
        self.original_keypoint_subscriber.stop()
        self.original_keypoint_subscriber_left.stop()
        self.transformed_keypoint_publisher.stop()

        print('Stopping the keypoint position transform process.')