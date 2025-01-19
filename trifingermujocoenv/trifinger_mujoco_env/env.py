import pathlib
from time import sleep, time
from typing import Any, Dict, Optional, Tuple

from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from dm_control.mujoco import Physics
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
from numpy.typing import NDArray
import sys


class Marker:

    def __init__(self, properties_dict: Dict, id: Optional[int], viewer_scene: Optional[Any]):
        self.id = id
        self.viewer_scene = viewer_scene
        self.properties_dict = properties_dict

    def update(self, attribute: str, value: Any):
        self.properties_dict[attribute] = value
        if self.viewer_scene is not None:
            setattr(self.viewer_scene.geoms[self.id], attribute, value)


class TriFingerEnv(gym.Env[NDArray[np.float64], NDArray[np.float32]]):
    """Simulated TriFinger platform using Mujoco."""

    # maximum torque in Nm
    _max_torque = 0.397
    _safety_damping_coeff = np.array([0.09, 0.09, 0.05] * 3)
    _max_velocity_radps = 10.0
    n_fingers = 3
    n_joints = 9
    n_cameras = 3
    _finger_angles = [0, 120, 240]
    _finger_joint_names = [
        "finger_base_to_upper_joint_",
        "finger_middle_to_lower_joint_",
        "finger_upper_to_middle_joint_",
    ]
    # original resolution: 540x720
    camera_im_height = 270
    camera_im_width = 360

    def __init__(
        self,
        render_mode: Optional[str] = None,
        image_width: int = 720,
        image_height: int = 720,
        image_obs: bool = False,
        start_viewer: bool = False,
        episode_length: int = 1000,
        dt: float = 0.01,
        n_substeps: int = 2,
        objects: str = "cube",
        # TODO: delay in observation, obs action delay
    ):
        """Initialize the TriFinger environment.
        
        Args:
            render_mode: How to render the environment.  Valid values are:
                - "human": Render in MuJoCo viewer.
                - "rgb_array": Return RGB image as observation.
                - None: Do not render the environment.
            image_width: Width of the image if render_mode is "rgb_array".
            image_height: Height of the image if render_mode is "rgb_array".
            image_obs: Include camera images in the observation.
            start_viewer: Start viewer even when not rendering in human mode.
            episode_length: Number of time steps in one episode.
            dt: Simulator time step for mujoco.
            n_substeps: How often to step the simulator per environment step.
            objects: Which objects to insert innto the simulation. Valid values are:
                - "cube": Use two cubes.
                - "two_cubers": Use two cubes.
                - "no_objects": Use no objects.
        """
        self.render_mode = render_mode
        self.image_width = image_width
        self.image_height = image_height
        self.image_obs = image_obs
        self.start_viewer = start_viewer
        self.episode_length = episode_length
        self.dt = dt
        self.n_substeps = n_substeps

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(9,))

        # observation space
        finger_pos_low = np.array([-0.33, 0.0, -2.7] * self.n_fingers, dtype=np.float32)
        finger_pos_high = np.array([1.0, 1.57, 0.0] * self.n_fingers, dtype=np.float32)
        finger_vel_low = np.full(self.n_joints, -self._max_velocity_radps, dtype=np.float32)
        finger_vel_high = np.full(self.n_joints, self._max_velocity_radps, dtype=np.float32)
        finger_pos_space = spaces.Box(low=finger_pos_low, high=finger_pos_high)
        finger_vel_space = spaces.Box(low=finger_vel_low, high=finger_vel_high)
        robot_space = spaces.Dict(
            {
                "position": finger_pos_space,
                "velocity": finger_vel_space,
            }
        )
        spaces_dict = {
            "robot": robot_space,
        }
        if self.image_obs:
            spaces_dict["camera_images"] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(
                    self.n_cameras,
                    self.camera_im_height,
                    self.camera_im_width,
                    3,
                ),
                dtype=np.uint8,
            )

        # NOTE: The order of dictionary items matters as it determines how
        # the observations are flattened/unflattened. The observation space
        # is therefore sorted by key.
        def sort_by_key(d):
            return {
                k: (
                    gym.spaces.Dict(sort_by_key(v.spaces))
                    if isinstance(v, gym.spaces.Dict)
                    else v
                )
                for k, v in sorted(d.items(), key=lambda item: item[0])
            }

        self.observation_space = spaces.Dict(
            # sort to make sure the order is not easily changed by changes to the code
            sort_by_key(spaces_dict)
        )

        code_dir = pathlib.Path(__file__).resolve().parent
        xml_path = (code_dir / f"assets/trifinger_with_{objects}.xml").as_posix()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # dm control suite Physics object for inverseke kinematics
        self._dm_physics = Physics.from_xml_path(xml_path)

        self.model.opt.timestep = self.dt

        if self.render_mode == "human" or self.start_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(
                self.model, height=self.image_height, width=self.image_width
            )
        
        self.camera_renderer = mujoco.Renderer(
            self.model,
            height=self.camera_im_height,
            width=self.camera_im_width
        ) 
        self._markers = []
        self._marker_visibilities = []

        self._last_frame_time = None
        self._steps = 0
        self.reset()

    def add_marker(self, marker: dict, visible_to_cameras: bool = False) -> None:
        """Add a marker to the scene.
        
        Args:
            marker: Dictionary with marker parameters. some of the
                supported values are:
                - pos: Position of the marker in world coordinates.
                - rgba: Color of the marker as RGBA values in [0, 1].
                - type: Type of the marker. See mjtGeom for possible values.
                - size: Size of the marker.
            visible_to_cameras: Whether the marker should be visible to the
                cameras. They are always visible in the viewer.
                
        Returns:
            The added marker (same dictionary as argument marker).
            Modifying this dictionary will update the marker in the scene.
        """
        self._markers.append(marker)
        self._marker_visibilities.append(visible_to_cameras)
        if self.render_mode == "human" or self.start_viewer:
            id = self._add_marker(marker, self.viewer.user_scn)
            viewer_scene = self.viewer.user_scn
        else:
            id = None
            viewer_scene = None
        return Marker(properties_dict=marker, id=id, viewer_scene=viewer_scene)

    def _add_marker(self, marker: dict, scene: Any) -> None:
        """Add a marker to the scene."""

        g = scene.geoms[scene.ngeom]
        # default values.
        g.dataid = -1
        g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        g.objid = -1
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        g.texid = -1
        g.texuniform = 0
        g.texrepeat[0] = 1
        g.texrepeat[1] = 1
        g.emission = 0
        g.specular = 0.5
        g.shininess = 0.5
        g.reflectance = 0
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size[:] = np.ones(3) * 0.1
        g.mat[:] = np.eye(3)
        g.rgba[:] = np.ones(4)

        for key, value in marker.items():
            if isinstance(value, (int, float, mujoco._enums.mjtGeom)):
                setattr(g, key, value)
            elif isinstance(value, (tuple, list, np.ndarray)):
                attr = getattr(g, key)
                attr[:] = np.asarray(value).reshape(attr.shape)
            elif isinstance(value, str):
                assert key == "label", "Only label is a string in mjtGeom."
                if value is None:
                    g.label[0] = 0
                else:
                    g.label = value
            elif hasattr(g, key):
                raise ValueError(
                    "mjtGeom has attr {} but type {} is invalid".format(
                        key, type(value)
                    )
                )
            else:
                raise ValueError("mjtGeom doesn't have field %s" % key)

        scene.ngeom += 1
        return scene.ngeom - 1

    def get_observation(self, detail: bool = True, state: bool = False) -> Dict:
        """Get the current observation.
        
        Returns:
            Dictionary with the current observation. See definition of the
            observation space for details."""
        obs = dict()
        if state:
            sim_obs = {
                "qpos": self.data.qpos,
                "qvel": self.data.qvel,
                "ctrl": self.data.ctrl,
            }
            obs["sim_state"] = sim_obs
        else:
            if self.image_obs:
                camera_images = self.get_camera_images()
                obs["camera_images"] = camera_images
        robot_obs = {
            "position": self.data.qpos[:9],
            "velocity": self.data.qvel[:9],
        }
        obs["robot"] = robot_obs
        return obs

    def get_camera_images(self) -> NDArray[np.uint8]:
        """Render camera images.
        
        Returns:
            3x270x360x3 array with the camera images. First dimension
            corresponds to the camera index.
        """
        images = np.full(
            (self.n_cameras, self.camera_im_height, self.camera_im_width, 3),
            0.0,
            dtype=np.uint8
        )
        for i in range(self.n_cameras):
            name = f"dataset_camera{i + 1}"
            self.camera_renderer.update_scene(self.data, camera=name)
            for marker_params, marker_vis in zip(self._markers, self._marker_visibilities):
                if marker_vis:
                    self._add_marker(marker_params, self.camera_renderer.scene)
            images[i:, ...] = self.camera_renderer.render()
        return images

    def get_tip_position(self) -> NDArray[np.float32]:
        """Get the position of the finger tips.
        
        Returns:
            3x3 array with the positions of the finger tips.
            First dimension corresponds to the finger index.
            The order of the fingers is [0, 120, 240]."""
        tip_positions = np.zeros((self.n_fingers, 3), dtype=np.float32)
        for i, alpha in enumerate(self._finger_angles):
            tip_positions[i] = self.data.geom(f"tip_{alpha}").xpos
        return tip_positions

    def get_tip_velocity(self) -> NDArray[np.float32]:
        """Get the velocity of the finger tips.
        
        Returns:
            3x3 array with the velocity of the finger tips.
            First dimension corresponds to the finger index.
            The order of the fingers is [0, 120, 240]."""
        tip_velocities = np.zeros((self.n_fingers, 3), dtype=np.float32)
        for i, alpha in enumerate(self._finger_angles):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"tip_{alpha}") 
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_GEOM, body_id, vel, True)
            tip_velocities[i] = vel[:3]
        return tip_velocities

    def get_angles_from_tip_pos(self, target_tip_pos: NDArray[np.float32]) -> NDArray[np.float32]:
        """Get the angles of the fingers from the tip positions (inverse kinematics).
        
        Args:
            tip_positions: 3x3 array with the positions of the finger tips.
        
        Returns:
            3x3 array with the angles of the fingers.
            First dimension corresponds to the finger index.
            The order of the fingers is [0, 120, 240]."""
        joint_angles = np.zeros((self.n_fingers, 3), dtype=np.float32)
        self._dm_physics.data.qpos[:9] = self.data.qpos[:9]
        for i, angle in enumerate(self._finger_angles):
            joint_names = [f"{name}{angle}" for name in self._finger_joint_names]
            result = qpos_from_site_pose(
                physics=self._dm_physics,
                site_name=f"tip_{angle}",
                target_pos=target_tip_pos[i],
                joint_names=joint_names,
                tol=1e-3,
                max_steps=10,
                # important to avoid considerable copying overhead
                inplace=True,
            )
            joint_angles[i] = result.qpos[3 * i : 3 * (i + 1)]
        return joint_angles

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[NDArray[np.float32], Dict]:
        """Reset the environment to the initial state.

        Args:
            seed: Seed to use for the random number generator. If not given, a
                random seed is used.
            options: Optional dictionary with additional options.
        Returns:
            Initial observation and info dict.
        """
        self._last_frame_time = 0.0
        self._steps = 0
        mujoco.mj_resetData(self.model, self.data)
        # initial finger position
        self.data.qpos[:9] = np.array([0.0, 0.9, -2.0] * 3)
        # needed for rendering to work
        mujoco.mj_forward(self.model, self.data)
        obs = self.get_observation(detail=True, state=True)
        return obs, {}

    def _safety_damping(self, torques: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply safety damping to the action.
        
        Args:
            torques: Raw torques (in Nm) to apply safety damping to.
        Returns:
            Action with safety damping applied (in Nm)."""
        torques = np.clip(torques, -self._max_torque, self._max_torque)
        return torques - self._safety_damping_coeff * self.data.qvel[:9]

    def render(self) -> Optional[np.ndarray]:
        """Render the environment.
        
        Returns:
            RGB image of the rendered scene."""
        if self.render_mode == "rgb_array":
            self.renderer.update_scene(self.data, camera="rgb_camera")
            for marker_params in self._markers:
                self._add_marker(marker_params, self.renderer.scene)
            return self.renderer.render()
        else:
            return None
    
    def gravity_compensation(self):
        gravity_vector = np.array(self.model.opt.gravity)
        for section in ["upper", "middle", "lower"]:
            for i, alpha in enumerate(self._finger_angles):
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"finger_{section}_link_{alpha}")
                body_mass = self.model.body_mass[body_id]
                gravity_force = body_mass * gravity_vector
                self.data.xfrc_applied[body_id][:3] = -gravity_force

    def step(
        self, action: NDArray[np.float32], return_obs: bool = False
    ) -> Tuple[NDArray[np.float32], np.float32, bool, bool, Dict[str, np.float64]]:
        """Execute one time step in the environment.
        
        Args:
            action: Action to apply to the environment.
        
        Returns:
            Observation, reward, terminated, truncated, info."""
        for _ in range(self.n_substeps):
            torques = action * self._max_torque
            torques = self._safety_damping(torques)
            self.data.ctrl = torques
            self.gravity_compensation()
            mujoco.mj_step(self.model, self.data)

        if return_obs:
            obs = self.get_observation(detail=True, state=True)
        else:
            obs = None

        if self.render_mode == "human" or self.start_viewer:
            self.viewer.sync()
            now_time = time()
            if now_time - self._last_frame_time < self.n_substeps * self.dt:
                sleep(self.n_substeps * self.dt - (now_time - self._last_frame_time))
            self._last_frame_time = now_time
        self._steps += 1
        truncated = self._steps >= self.episode_length
        return obs, 0.0, False, truncated, {}