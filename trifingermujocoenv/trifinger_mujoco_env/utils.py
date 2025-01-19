"""Utility functions related to poses, keypoints and sampling of poses (in particular for a cube)."""

import numpy as np
from scipy.spatial.transform import Rotation 

_CUBE_WIDTH = 0.065
_ARENA_RADIUS = 0.195

_cube_3d_radius = np.sqrt(3) * _CUBE_WIDTH / 2
_max_height = 0.1


def to_mujoco_quat(quat):
    return np.array([quat[3], quat[0], quat[1], quat[2]])


def to_scipy_quat(quat):
    return np.array([quat[1], quat[2], quat[3], quat[0]])


def random_yaw_orientation():
    # which axis should be oriented vertically
    vert_axis = np.random.randint(0, 3)
    # should it be pointing up or down
    flip = np.random.choice([-1.0, 1.0])
    # construct a rotation matrix that aligns the vertical axis with the
    # chosen axis and points in the correct direction
    rot_mat = np.eye(3)
    rot_mat[:, 2] = rot_mat[:, vert_axis].copy()
    rot_mat[:, vert_axis] = flip * np.array([0, 0, 1])
    up_face_rot = Rotation.from_matrix(rot_mat)
    # then draw a random yaw rotation
    yaw_angle = np.random.uniform(0, 2 * np.pi)
    yaw_rot = Rotation.from_euler("z", yaw_angle)
    # and combine them
    orientation = yaw_rot * up_face_rot
    return yaw_angle, orientation.as_quat()


def random_xy(cube_yaw: float):
    """Sample an xy position for cube which maximally covers arena. 

    In particular, the cube can touch the barrier for all yaw angles."""

    theta = np.random.uniform(0, 2*np.pi)

    # Minimum distance of cube center from arena boundary
    min_dist = _CUBE_WIDTH/np.sqrt(2)*max(abs(np.sin(0.25*np.pi + cube_yaw - theta)), abs(np.cos(0.25*np.pi + cube_yaw - theta)))

    # sample uniform position in circle
    # (https://stackoverflow.com/a/50746409)
    radius = 0.1 * np.sqrt(np.random.random()) #(_ARENA_RADIUS - min_dist) * np.sqrt(np.random.random())

    # x,y-position of the cube
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    return x, y


def sample_initial_cube_pose():
    yaw_angle, orientation = random_yaw_orientation()
    x, y = random_xy(yaw_angle)
    z = _CUBE_WIDTH / 2
    position = np.array((x, y, z))
    orientation = to_mujoco_quat(orientation)
    return position, orientation


def sample_goal_cube_pose(task_id: int):
    assert task_id in [0, 1, 2, 4], "Invalid task_id"
    if task_id == 0 or task_id == 1:
        position, orientation = sample_initial_cube_pose()
        rot_matrix = Rotation.from_quat(to_scipy_quat(orientation))
        # push task
        return position, orientation, rot_matrix.as_matrix()
    elif task_id == 4 or task_id == 2:
        # lift task
        x, y = random_xy(0.0)
        z = np.random.uniform(_cube_3d_radius, _max_height)
        position = np.array((x, y, z))
        rot_matrix = Rotation.random()
        orientation = to_mujoco_quat(rot_matrix.as_quat())
        return position, orientation, rot_matrix.as_matrix()


def to_pybullet_quat(x):
    return np.array([x.x, x.y, x.z, x.w])


def to_numpy_quat(x):
    return np.quaternion(x[3], x[0], x[1], x[2])


def to_world_space(x_local, position, orientation):
    q_rot = orientation
    q_local = np.quaternion(0., x_local[0], x_local[1], x_local[2])
    q_global = q_rot * q_local * q_rot.conjugate()
    return position + np.array([q_global.x, q_global.y, q_global.z])


def get_keypoints_from_pose(position, orientation, num_keypoints=8, dimensions=(0.065, 0.065, 0.065)):
    keypoints = []
    for i in range(num_keypoints):
        # convert to binary representation
        str_kp = "{:03b}".format(i)
        # set components of keypoints according to digits in binary representation
        loc_kp = [(1. if str_kp[i] == "0" else -1.) * 0.5 * d for i, d in enumerate(dimensions)][::-1]
        glob_kp = to_world_space(loc_kp, position, orientation)
        keypoints.append(glob_kp)

    return np.array(keypoints, dtype=np.float32)