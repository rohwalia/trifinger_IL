import os
import time
import h5py
import hydra
import numpy as np
from .recorder import Recorder
from openteach.utils.timer import FrequencyTimer
from openteach.constants import *
from openteach.utils.network import ZMQKeypointSubscriber, ZMQKeypointPublisher 


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

# To record robot information
class SimInformationRecord(Recorder):
    def __init__(
        self,
        port_configs,
        recorder_function_key,
        storage_path,
    ):

        # Port configs
        self.robot = port_configs['robot']
        host = port_configs['host']
        observation_port = port_configs["observation_publish_port"]
        # jointanglesubscribeport = port_configs['jointanglesubscribeport']
        # actualjointanglesubscribeport = port_configs['actualjointanglesubscribeport']
        # timestampsubscribeport = port_configs['timestampssubscribeport']
        # endeff_publish_port= port_configs['endeff_publish_port']
        # endeffpossubscribeport= port_configs['endeffpossubscribeport']
        
        self.subscriber =  ZMQKeypointSubscriber(
            host = host,
            port = observation_port,
            topic = "observation"
        )

        # Subscriber for desired angles
        # self.subscriber =  ZMQKeypointSubscriber(
        #     host = host,
        #     port = jointanglesubscribeport,
        #     topic= 'desired_angles'
        # )

        # # Subscriber for Joint Angles
        # self.jointanglesubscriber =  ZMQKeypointSubscriber(
        #     host = host,
        #     port = actualjointanglesubscribeport,
        #     topic= 'current_angles'
        # )
        # # Subscriber for Timestamps
        # self.timestampsubscriber =  ZMQKeypointSubscriber(
        #     host = host,
        #     port = timestampsubscribeport,
        #     topic= 'timestamps'
        # )

        # # Subscriber for Actual End effector position
        # self.end_eff_coords_actual =  ZMQKeypointSubscriber(
        #     host = host,
        #     port = endeff_publish_port,
        #     topic= 'endeff_coords'
        # ) 

        # # Publisher for End effector position
        # self.endeffector_pos_subscriber =  ZMQKeypointSubscriber(
        #         host = host,
        #         port = endeffpossubscribeport,
        #         topic= 'endeff_coords'
        #     )
        
    

        
        self.timer = FrequencyTimer(VR_FREQ)
        # self.timestampsubscribeport= timestampsubscribeport
        self.recorder_function_key = recorder_function_key
        # Storage path for file
        self._filename = '{}_{}'.format(self.robot, recorder_function_key)
        self.notify_component_start('{}'.format(self._filename))
        self._recorder_file_name = os.path.join(storage_path, self._filename + '.h5')

        # Initializing data containers
        self.robot_information = dict()

    def stream(self):
        print('Checking if the keypoint port is active...')

        print('Starting to record keypoints to store in {}.'.format(self._recorder_file_name))

        self.num_datapoints = 0
        self.record_start_time = time.time()

        while True:
            self.timer.start_loop()
            try:
                if self.robot == "trifinger_sim":
                    if self.recorder_function_key == "observation":
                        start = time.time()
                        observation_states = self.subscriber.recv_keypoints()
                        #print("Entering Observation States")
                        print("Time for data collection:", time.time()-start)
                        if not self.robot_information:
                            self.keys = get_nested_keys(observation_states)
                        for key in self.keys:
                            if key not in self.robot_information.keys():
                                self.robot_information[key] = [get_value_from_nested_dict(observation_states, key)]
                                # if key =='timestamp':
                                #     self.robot_information[key]=[observation_states["timestamp"]]
                                # elif key =='robot_position':
                                #     self.robot_information[key]=[observation_states["robot"]["position"]]
                                # elif key =='robot_velocity':
                                #     self.robot_information[key]=[observation_states["robot"]["velocity"]]
                                # elif key =='images':
                                #     self.robot_information[key]=[observation_states["camera_images"]]          
                            else:
                                self.robot_information[key].append(get_value_from_nested_dict(observation_states, key))
                                # if key =='timestamp':
                                #     self.robot_information[key].append(observation_states["timestamp"])
                                # elif key =='robot_position':
                                #     self.robot_information[key].append(observation_states["robot"]["position"])
                                # elif key =='robot_velocity':
                                #     self.robot_information[key].append(observation_states["robot"]["velocity"])
                                # elif key =='images':
                                #     self.robot_information[key].append(observation_states["camera_images"]) 
                # timestamps= self.timestampsubscriber.recv_keypoints()
                # if self.robot=='allegro':
                #     if self.recorder_function_key =='joint_states':
                #         actual_joint_angles=self.jointanglesubscriber.recv_keypoints()
                #     else:
                #         commanded_joint_angles=self.subscriber.recv_keypoints()
                # elif self.robot=='moving_allegro' or self.robot=='franka_allegro':
                #     if self.recorder_function_key =='joint_states':
                #         actual_joint_angles=self.jointanglesubscriber.recv_keypoints()
                #         print("Entering joint states")
                #     elif self.recorder_function_key=='cartesian_states':
                #         actual_endeff_coords=self.end_eff_coords_actual.recv_keypoints()
                #         print("Entering cartesian states")
                #     elif self.recorder_function_key=='commanded_cartesian_states':
                #         commanded_endeff_coords=self.endeffector_pos_subscriber.recv_keypoints()
                #         print("Entering Commanded Cartesian States")
                #     else:
                #         commanded_joint_angles=self.subscriber.recv_keypoints()
                #         print("Entering Commanded Joint States")
                # else:
                #     if self.recorder_function_key=='cartesian_states':
                #         actual_endeff_coords=self.end_eff_coords_actual.recv_keypoints()
                #     else:
                #         actual_endeff_coords=self.endeffector_pos_subscriber.recv_keypoints()
               
                # # timestamps= self.timestampsubscriber.recv_keypoints()
                # # proprio
                # for key in ['position', 'timestamps']:  
                #     if self.robot=='allegro':
                #         if self.recorder_function_key == 'joint_states':
                #             if key not in self.robot_information.keys():
                #                 if key =='timestamps':
                #                     self.robot_information[key]=[timestamps]
                #                 elif key =='position':
                #                     self.robot_information[key]=[actual_joint_angles]   
                #             else:
                #                 if key =='timestamps':
                #                     self.robot_information[key].append(timestamps)
                #                 elif key =='position':
                #                     self.robot_information[key].append(actual_joint_angles)
                #         elif self.recorder_function_key == 'commanded_joint_states':
                #             if key not in self.robot_information.keys():
                #                 if key =='timestamps':
                #                     self.robot_information[key]=[timestamps]
                #                 elif key =='position':
                #                     self.robot_information[key]=[commanded_joint_angles]   
                #             else:
                #                 if key =='timestamps':
                #                     self.robot_information[key].append(timestamps)
                #                 elif key =='position':
                #                     self.robot_information[key].append(commanded_joint_angles)
                #     elif self.robot=='moving_allegro' or self.robot== 'franka_allegro':
                #         if self.recorder_function_key == 'joint_states':
                #             if key not in self.robot_information.keys():
                #                 if key =='timestamps':
                #                     self.robot_information[key]=[timestamps]
                #                 elif key =='position':
                #                     self.robot_information[key]=[actual_joint_angles]   
                #             else:
                #                 if key =='timestamps':
                #                     self.robot_information[key].append(timestamps)
                #                 elif key =='position':
                #                     self.robot_information[key].append(actual_joint_angles)
                #             print('Joint angles recorded', actual_joint_angles)

                #         elif self.recorder_function_key == 'commanded_joint_states':
                #             if key not in self.robot_information.keys():
                #                 if key =='timestamps':
                #                     self.robot_information[key]=[timestamps]
                #                 elif key =='position':
                #                     self.robot_information[key]=[commanded_joint_angles]   
                #             else:
                #                 if key =='timestamps':
                #                     self.robot_information[key].append(timestamps)
                #                 elif key =='position':
                #                     self.robot_information[key].append(commanded_joint_angles)

                #         elif self.recorder_function_key == 'commanded_cartesian_states':
                #             if key not in self.robot_information.keys():
                #                 if key =='timestamps':
                #                     self.robot_information[key]=[timestamps]
                #                 elif key =='position':
                #                     self.robot_information[key]=[commanded_endeff_coords]   
                #             else:
                #                 if key =='timestamps':
                #                     self.robot_information[key].append(timestamps)
                #                 elif key =='position':
                #                     self.robot_information[key].append(commanded_endeff_coords)
                        
                #         else:
                #             if key not in self.robot_information.keys():
                #                 if key =='timestamps':
                #                     self.robot_information[key]=[timestamps]
                #                 elif key =='position':
                #                     self.robot_information[key]=[actual_endeff_coords]   

                #             else:
                #                 if key =='timestamps':
                #                     self.robot_information[key].append(timestamps)
                #                 elif key =='position':
                #                     self.robot_information[key].append(actual_endeff_coords)
                #     else:
                #         if self.recorder_function_key == 'cartesian_states':
                #             if key not in self.robot_information.keys():
                #                 if key =='timestamps':
                #                     self.robot_information[key]=[timestamps]
                #                 elif key =='position':
                #                     self.robot_information[key]=[actual_endeff_coords]   
                #             else:
                #                 if key =='timestamps':
                #                     self.robot_information[key].append(timestamps)
                #                 elif key =='position':
                #                     self.robot_information[key].append(actual_endeff_coords)
                                
                #         else:
                #             if key not in self.robot_information.keys():
                #                 if key =='timestamps':
                #                     self.robot_information[key]=[timestamps]
                #                 elif key =='position':
                #                     self.robot_information[key]=[commanded_endeff_coords]   
                #             else:
                #                 if key =='timestamps':
                #                     self.robot_information[key].append(timestamps)
                #                 elif key =='position':
                #                     self.robot_information[key].append(commanded_endeff_coords)


                self.num_datapoints += 1
                self.timer.end_loop()
            except KeyboardInterrupt:
                self.record_end_time = time.time()
                break

        # Displaying statistics
        self._display_statistics(self.num_datapoints)
        
        # Saving the metadata
        self._add_metadata(self.num_datapoints)

        # Writing to dataset
        print('Compressing keypoint data...')
        with h5py.File(self._recorder_file_name, "w") as file:
            for key in self.robot_information.keys():
                if key != 'timestamp':
                    print("Keys saving", key)
                    self.robot_information[key] = np.array(self.robot_information[key], dtype = np.float32)
                    file.create_dataset(key, data = self.robot_information[key], compression="gzip", compression_opts = 6)
                else:
                    self.robot_information['timestamp'] = np.array(self.robot_information['timestamp'], dtype = np.float64)
                    file.create_dataset('timestamp', data = self.robot_information['timestamp'], compression="gzip", compression_opts = 6)

            # Other metadata
            file.update(self.metadata)
        print('Saved keypoint data in {}.'.format(self._recorder_file_name))