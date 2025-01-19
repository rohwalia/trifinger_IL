from gymnasium import spaces

import numpy as np
import quaternion
from typing import Dict

import mujoco
from scipy.spatial.transform import Rotation 

from .env import TriFingerEnv
from .utils import sample_initial_cube_pose, sample_goal_cube_pose, get_keypoints_from_pose, to_scipy_quat


class MoveCubeEnv(TriFingerEnv):
    """Environment for moving a cube with the TriFinger robot."""

    _task_ids = {
        "push_reorient": 0,
        "lift_reorient": 4,
        "push": 1,
        "lift": 2
    }
    n_keypoints = 8
    # parameters of reward function
    _kernel_reward_weight = 4.0
    _logkern_scale = 30
    _logkern_offset = 2

    _position_threshold = 0.02
    _angle_threshold_deg = 22.0

    def __init__(self, task: str = "lift", fix_task: bool = False, **kwargs):
        """Initialize.
        
        Args:
            task: Task to be solved.  Currently, the following tasks are
                supported:
                - "push_reorient": Push the cube to a target position and orientation on the floor.
                    Note that this task is different from the push task in the TriFingerRLDatasets and
                    the Real Robot Competitions as they only required matching a target position.
                - "lift": Lift the cube to a target position in the air and
                    match a target pose.
            kwargs: Keyword arguments that are passed to the superclass.
        """
        assert task in self._task_ids, f"Task {task} not found in {self._task_ids}"
        self.task = task
        self.fix_task = fix_task
        self.task_id = self._task_ids[task]
        self.goal_marker = None
        self._sample_goal()
        super().__init__(objects="cube", **kwargs)

        if self.goal_marker is None:
            # goal marker
            if self.task_id == 1 or self.task_id == 2:
                self.goal_marker = self.add_marker(
                    {
                        "pos": [0.0, 0.0, 0.0],
                        "rgba": [1.0, 1.0, 1.0, 0.3],
                        "type": 2,
                        "size": [0.0325, 0.0325, 0.0325],
                    },
                    visible_to_cameras=True,
                )
            else:
                texture_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TEXTURE, "cube_texture")
                self.goal_marker = self.add_marker(
                    {
                        "pos": [0.0, 0.0, 0.0],
                        "rgba": [1.0, 1.0, 1.0, 0.3],
                        "type": 6,
                        "size": [0.0325, 0.0325, 0.0325],
                        "texid": texture_id,
                    },
                    visible_to_cameras=True,
                )

        keypoints_space = spaces.Box(
            low=np.array([[-0.6, -0.6, 0.0]] * self.n_keypoints),
            high=np.array([[0.6, 0.6, 0.3]] * self.n_keypoints),
            dtype=np.float32,
        )
        cube_space = spaces.Dict({
            "position": spaces.Box(
                low=np.array([-0.6, -0.6, 0.0]),
                high=np.array([0.6, 0.6, 0.3]),
                dtype=np.float32,
            ),
            "orientation": spaces.Box(
                low=np.ones(4) * -1.0,
                high=np.ones(4),
                dtype=np.float32,
            ),
        })

        self.observation_space = {
            **self.observation_space,
            "achieved_goal": {
                "keypoints": keypoints_space,
            },
            "desired_goal": {
                "keypoints": keypoints_space,
            },
            "objects": {
                "cube": cube_space, 
            }
        }

    def _sample_initial_pose(self):
        """Sample a new initial pose for the cube.
        
        The cube is placed on the floor and rotated randomly."""
        # if self.fix_task:
        #     position, orientation = np.array((0.14, 0.0, 0.065/2)), np.array([0,1,0,0])
        # else:
        position, orientation = sample_initial_cube_pose()
        cube_joint_index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        self.data.qpos[cube_joint_index:cube_joint_index + 3] = position
        self.data.qpos[cube_joint_index + 3:cube_joint_index + 7] = orientation

    def _sample_goal(self):
        """Sample new goal position and orientation for the cube.
        
        The goal distribution depends on the chosen task."""
        self.goal_position, self.goal_orientation, self.goal_mat = sample_goal_cube_pose(self.task_id)
        if self.goal_marker is not None:
            if self.task_id == 1 or self.task_id == 2:
                if self.fix_task:
                    if self.task_id==1:
                        self.goal_position = np.array((0.0, 0.0, 0.065/2))
                    else:
                        self.goal_position = np.array((0.0, 0.0, 0.085))
                self.goal_marker.update("pos", self.goal_position)
            else:
                if self.fix_task:
                    self.goal_position = np.array((0.0, 0.0, 0.065/2))
                    self.goal_orientation = np.array([[1,0,0], [0,1,0], [0,0,1]])
                self.goal_marker.update("pos", self.goal_position)
                self.goal_marker.update("mat", self.goal_mat)

    def set_goal(self, position, orientation):
        """Set a new goal position and orientation for the cube.

        Args:
            position: Goal position of the cube.
            orientation: Goal orientation of the cube.
        """
        self.goal_position = position
        self.goal_orientation = orientation
        self.goal_mat = Rotation.from_quat(to_scipy_quat(orientation)).as_matrix()
        if self.goal_marker is not None:
            if self.task_id == 1 or self.task_id == 2:
                self.goal_marker.update("pos", self.goal_position)
            else:
                self.goal_marker.update("pos", self.goal_position)
                self.goal_marker.update("mat", self.goal_mat)

    def _kernel_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray
    ) -> float:
        """Compute reward by evaluating a logistic kernel on the pairwise distance of
        points.

        Parameters can be either a 1 dim. array of size 3 (positions) or a two dim.
        array with last dim. of size 3 (keypoints)

        Args:
            achieved_goal: Position or keypoints of current pose of the object.
            desired_goal: Position or keypoints of goal pose of the object.
        """
        if self.task_id == 1 or self.task_id == 2:
            achieved_goal = np.mean(achieved_goal)
            desired_goal = np.mean(desired_goal)
        diff = achieved_goal - desired_goal
        if diff.ndim > 1:
            dist = np.linalg.norm(diff, axis=-1)
        else:
            dist = np.linalg.norm(diff)
        scaled = self._logkern_scale * dist
        # Use logistic kernel
        rew = self._kernel_reward_weight * np.mean(
            1.0 / (np.exp(scaled) + self._logkern_offset + np.exp(-scaled))
        )
        return rew

    def _has_achieved(
        self,
        achieved_pos: np.ndarray,
        achieved_quat: np.quaternion,
        desired_pos: np.ndarray,
        desired_quat: np.quaternion,
    ) -> bool:
        """Determine whether goal pose is achieved."""
        position_diff = np.linalg.norm(
            desired_pos - achieved_pos
        )
        # cast from np.bool_ to bool to make mypy happy
        position_check = bool(position_diff < self._position_threshold)

        if self.task_id==1 or self.task_id == 2:
            return position_check

        a = desired_quat
        b = achieved_quat
        b_conj = b.conjugate()
        quat_prod = a * b_conj
        norm = np.linalg.norm([quat_prod.x, quat_prod.y, quat_prod.z])
        norm = min(norm, 1.0)  # type: ignore
        angle = 2.0 * np.arcsin(norm)
        orientation_check = angle < 2.0 * np.pi * self._angle_threshold_deg / 360.0

        return position_check and orientation_check

    def has_achieved(self) -> bool:
        """Determine whether goal pose is achieved based ."""
        cube_joint_index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        achieved_pos = self.data.qpos[cube_joint_index:cube_joint_index + 3]
        orientation = self.data.qpos[cube_joint_index + 3:cube_joint_index + 7]
        achieved_quat = quaternion.from_float_array(orientation)
        desired_quat = quaternion.from_float_array(self.goal_orientation)
        return self._has_achieved(
            achieved_pos, achieved_quat, self.goal_position, desired_quat
        )
    
    def has_achieved_posthoc(self, achieved_goal: dict, desired_goal: dict) -> bool:
        """Determine whether goal pose is achieved from given achieved goal."""
        # TODO: Convert keypoints to position and orientation and check
        raise NotImplementedError

    def get_observation(self, detail: bool = True, state: bool = False) -> Dict:
        """Get the current observation.
        
        Returns:
            Dictionary with the current observation. See definition of the
            observation space for details."""
        obs = super().get_observation(detail, state)
        if detail:
            desired_keypoints = get_keypoints_from_pose(
                self.goal_position, quaternion.from_float_array(self.goal_orientation)
            ).flatten()
            cube_joint_index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
            position = self.data.qpos[cube_joint_index:cube_joint_index + 3]
            orientation = self.data.qpos[cube_joint_index + 3:cube_joint_index + 7]
            orientation_quant = quaternion.from_float_array(orientation)
            achieved_keypoints = get_keypoints_from_pose(position, orientation_quant).flatten()
            obs["achieved_goal"] = {
                "keypoints": achieved_keypoints,
            }
            obs["desired_goal"] = {
                "keypoints": desired_keypoints,
            }
            obs["objects"] = {
                "cube": {
                    "position": position,
                    "orientation": orientation,
                }
            }
        return obs

    def step(self, action, return_obs: bool = False):
        obs, _, terminated, truncated, info = super().step(action, return_obs)
        if return_obs:
            rew = self._kernel_reward(
                obs["achieved_goal"]["keypoints"], obs["desired_goal"]["keypoints"]
            )
        else:
            rew = None
        info["has_achieved"] = self.has_achieved()
        return obs, rew, terminated, truncated, info["has_achieved"]

    def reset(self):
        ret_vals = super().reset()
        self._sample_initial_pose()
        self._sample_goal()
        mujoco.mj_forward(self.model, self.data)
        return ret_vals