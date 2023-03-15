import numpy as np
import pyflex
import gym
from softgym.envs.cloth_env import FlexEnv
from softgym.action_space.action_space import PickerPickPlace
from softgym.utils.gemo_utils import *


class ClothEnv3D(FlexEnv):
    def __init__(
        self,
        particle_radius=0.00625,
        picker_radius=0.005,
        num_pickers=1,
        success_threshold=0.003,
        headless=False,
        **kwargs
    ):
        self.cloth_particle_radius = particle_radius
        self.num_pickers = num_pickers
        self.success_threshold = success_threshold
        self.headless = headless
        super().__init__(**kwargs)

        # cloth shape
        self.config = self.get_default_config()

        self.update_camera(
            self.config["camera_name"],
            self.config["camera_params"][self.config["camera_name"]],
        )
        self.unscaled_action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
        self.action_tool = PickerPickPlace(
            num_picker=self.num_pickers,
            picker_radius=picker_radius,
            particle_radius=particle_radius,
            env=self,
            picker_low=(-0.6, 0.0125, -0.6),
            picker_high=(0.6, 0.6, 0.6),
            collect_steps=False,
        )
        self.action_tool.delta_move = 0.005
        self.goal_pcd_points = None
        self.reset_act = np.array([0.0, 0.1, -0.6, 0.0, 0.0, 0.1, -0.6, 0.0])
        self.reset_pos = np.array([0.0, 0.1, -0.6, 0.0, 0.1, -0.6])

    def get_default_config(self):
        particle_radius = self.cloth_particle_radius
        cam_pos, cam_angle = np.array([-0.0, 0.65, 0.0]), np.array(
            [0, -np.pi / 2.0, 0.0]
        )
        config = {
            "ClothPos": [-0.15, 0.0, -0.15],
            "ClothSize": [int(0.30 / particle_radius), int(0.30 / particle_radius)],
            "ClothStiff": [
                2.0,
                0.5,
                1.0,
            ],  # Stretch, Bend and Shear #0.8, 1., 0.9 #1.0, 1.3, 0.9
            "mass": 0.0054,
            "camera_name": "default_camera",
            "camera_params": {
                "default_camera": {
                    "pos": cam_pos,
                    "angle": cam_angle,
                    "width": 720,
                    "height": 720,
                }
            },
            "flip_mesh": 0,
            "drop_height": 0.0,
        }

        return config

    def _get_flat_pos(self):
        dimx, dimy = self.config["ClothSize"]
        x = np.array([i * self.cloth_particle_radius for i in range(dimx)])
        y = np.array([i * self.cloth_particle_radius for i in range(dimy)])
        x = x - np.mean(x)
        y = y - np.mean(y)
        xx, yy = np.meshgrid(x, y)

        curr_pos = np.zeros([dimx * dimy, 3], dtype=np.float32)
        curr_pos[:, 0] = xx.flatten()
        curr_pos[:, 2] = yy.flatten()
        curr_pos[:, 1] = 5e-3  # Set specifally for particle radius of 0.00625
        return curr_pos

    def _set_to_flat(self):
        curr_pos = pyflex.get_positions().reshape((-1, 4))
        flat_pos = self._get_flat_pos()
        curr_pos[:, :3] = flat_pos
        pyflex.set_positions(curr_pos)
        pyflex.step()

    def _set_picker_pos(self, picker_pos):
        picker_pos = np.reshape(picker_pos, [-1, 3])
        shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        for i in range(shape_states.shape[0]):
            shape_states[i, 3:6] = picker_pos[i]
            shape_states[i, :3] = picker_pos[i]
        pyflex.set_shape_states(shape_states)

    def _get_rgbd(self, cloth_only=False):
        if cloth_only:
            rgb, depth = pyflex.render_cloth()
        else:
            rgb, depth = pyflex.render()
        rgb = np.array(rgb).reshape(
            self.camera_params["default_camera"]["height"],
            self.camera_params["default_camera"]["width"],
            4,
        )
        # rgb = rgb[::-1, :, :]  # reverse the height dimension
        rgb = rgb[:, :, :3]
        depth = np.array(depth).reshape(
            self.camera_params["default_camera"]["height"],
            self.camera_params["default_camera"]["width"],
        )
        # depth = depth[::-1, :]  # reverse the height dimension
        # depth[depth >= 999] = 0  # use 0 instead of 999 for null
        return rgb, depth

    def _get_cloud(self):
        return pyflex.get_positions().reshape(-1, 4)[:, :3]

    def get_observations(self, cloth_only=True):
        self.render(mode="rgb_array")
        rgb, depth = self._get_rgbd(cloth_only=cloth_only)
        object_pcd_points = self._get_cloud()
        obs = {
            "color": rgb, 
            "depth": depth, 
            "object_pcd_points": object_pcd_points,
            "goal_pcd_points": self.goal_pcd_points,
            # "poke_idx": 0 # TODO replace with predicted poke idx
        }

        # depth = depth * 255
        # depth = depth.astype(np.uint8)
        # depth_st = np.dstack([depth, depth, depth])
        # depth_img = Image.fromarray(depth_st)
        # depth_img = depth_img.resize(size=(200, 200))
        # depth_img = np.array(depth_img)
        # obs["depth"] = depth

        return obs

    def set_scene(self):
        render_mode = 2  # cloth
        env_idx = 0
        config = self.config
        camera_params = config["camera_params"][config["camera_name"]]
        scene_params = np.array(
            [
                *config["ClothPos"],
                *config["ClothSize"],
                *config["ClothStiff"],
                render_mode,
                *camera_params["pos"][:],
                *camera_params["angle"][:],
                camera_params["width"],
                camera_params["height"],
                config["mass"],
                config["flip_mesh"],
            ],
            dtype=np.float32,
        )
        pyflex.set_scene(env_idx, scene_params, 0)

    def reset(self, goal_pcd_points=None):
        self.set_scene()
        self.particle_num = pyflex.get_n_particles()
        self.prev_reward = 0.0
        self.time_step = 0
        self._set_to_flat()
        if hasattr(self, "action_tool"):
            self.action_tool.reset([0, 0.1, 0])
            self._set_picker_pos(self.reset_pos)
        # if self.recording:
            # self.video_frames.append(self.render(mode="rgb_array"))

        if goal_pcd_points is not None:
            self.goal_pcd_points = goal_pcd_points

        self.render(mode="rgb_array")
        obs = self.get_observations()

        return obs

    # def _compute_reward(self, curr_cloud):
    #     if self.goal_cloud is None:
    #         return -1000
    #     pos_metric = np.linalg.norm(self.goal - curr_cloud, axis=1).mean()
    #     return pos_metric

    def step(
        self,
        action,
        record_continuous_video=False,
        img_size=None,
        # pickplace=False,
        # on_table=False,
    ):
        """If record_continuous_video is set to True, will record an image for each sub-step"""
        # frames = []
        # obs = self.get_observations()
        for i in range(self.action_repeat):
            self._step(action)
            # self._step(action, pickplace, on_table=on_table)
            # if record_continuous_video and i % 2 == 0:  # No need to record each step
                # frames.append(self.get_image(img_size, img_size))
        nobs = self.get_observations(cloth_only=False)
        # reward = self._compute_reward(nobs['cloud'])
        # info = self._get_info()

        # if self.recording:
            # self.video_frames.append(self.render(mode="rgb_array"))
        self.time_step += 1

        # done = False
        # if self.time_step >= self.horizon:
            # done = True
        # if record_continuous_video:
            # info["flex_env_recorded_frames"] = frames
        # return nobs, reward, done, info
        return nobs

    # def _step(self, action, pickplace=False, on_table=True):
    def _step(self, action):
        """Action is the 3D coordinate of a cloth node and the flow of the action"""
        # TODO handle multiple pickers
        location, action_flow = action

        # Teleport picker to location and grasp
        self._set_picker_pos(location)
        grasp_action = np.concatenate([location, [1]], axis=0)
        self.action_tool.step(grasp_action, render=not self.headless)

        # TODO move to function
        max_scale = [0.25, 0.125, 0.25]
        action_scaled = action_flow.copy()
        action_scaled[0] *= max_scale[0]
        action_scaled[1] = ((action_scaled[1] + 1) / 2) * max_scale[1] # rescaled [-1, 1] to [0, max]
        action_scaled[2] *= max_scale[2]
        # action_scaled = action_flow * 0.1

        move_action = np.concatenate([location + action_scaled, [1]], axis=0)
        self.action_tool.step(move_action, render=not self.headless)

        drop_action = np.concatenate([location + action_scaled, [0]], axis=0)
        self.action_tool.step(drop_action, render=not self.headless)

        # go to neutral position
        self._set_picker_pos(self.reset_pos)
        for _ in range(50):
            self.action_tool.step(self.reset_act, render=not self.headless)

    def _rescale_action_flow(self, action_flow):
        """Rescale action flow from [-1, 1] in all dimensions"""


    def _get_info(self):
        return {}


# def uv_to_world_pos(camera_params, depth, u, v, particle_radius=0.0075, on_table=False):
#     height, width = depth.shape
#     K = intrinsic_from_fov(height, width, 45)  # the fov is 90 degrees

#     # from cam coord to world coord
#     cam_x, cam_y, cam_z = (
#         camera_params["default_camera"]["pos"][0],
#         camera_params["default_camera"]["pos"][1],
#         camera_params["default_camera"]["pos"][2],
#     )
#     cam_x_angle, cam_y_angle, cam_z_angle = (
#         camera_params["default_camera"]["angle"][0],
#         camera_params["default_camera"]["angle"][1],
#         camera_params["default_camera"]["angle"][2],
#     )

#     # get rotation matrix: from world to camera
#     matrix1 = get_rotation_matrix(-cam_x_angle, [0, 1, 0])
#     matrix2 = get_rotation_matrix(-cam_y_angle - np.pi, [1, 0, 0])
#     rotation_matrix = matrix2 @ matrix1

#     # get translation matrix: from world to camera
#     translation_matrix = np.eye(4)
#     translation_matrix[0][3] = -cam_x
#     translation_matrix[1][3] = -cam_y
#     translation_matrix[2][3] = -cam_z
#     matrix = np.linalg.inv(rotation_matrix @ translation_matrix)

#     x0 = K[0, 2]
#     y0 = K[1, 2]
#     fx = K[0, 0]
#     fy = K[1, 1]

#     z = depth[int(np.rint(u)), int(np.rint(v))]
#     if on_table or z == 0:
#         vec = ((v - x0) / fx, (u - y0) / fy)
#         z = (particle_radius - matrix[1, 3]) / (
#             vec[0] * matrix[1, 0] + vec[1] * matrix[1, 1] + matrix[1, 2]
#         )
#     else:
#         # adjust for particle radius from depth image
#         z -= particle_radius

#     x = (v - x0) * z / fx
#     y = (u - y0) * z / fy

#     cam_coord = np.ones(4)
#     cam_coord[:3] = (x, y, z)
#     world_coord = matrix @ cam_coord

#     return world_coord