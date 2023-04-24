import numpy as np
import pyflex
import copy
import random
from pathlib import Path
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
        record=False,
        starts_list=[],
        goals_list=[],
        wait_steps=20,
        planar_action=True,
        max_action_scale = [0.125, 0.125, 0.125],
        **kwargs
    ):
        self.cloth_particle_radius = particle_radius
        self.num_pickers = num_pickers
        self.success_threshold = success_threshold
        self.headless = headless
        self.record = record
        self.starts_list = starts_list
        self.goals_list = goals_list
        self.wait_steps = wait_steps
        self.planar_action = planar_action
        self.max_action_scale = max_action_scale # x z y 
        super().__init__(headless=headless, **kwargs)

        # cloth shape
        self.config = self.get_default_config()

        self.update_camera(
            self.config["camera_name"],
            self.config["camera_params"][self.config["camera_name"]],
        )
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
        self.reset_act = np.array([0.0, 0.1, -0.6, 0.0, 0.0, 0.1, -0.6, 0.0])
        self.reset_pos = np.array([0.0, 0.1, -0.6, 0.0, 0.1, -0.6])

        self.goal_pcd_points = None

    def _sample_cloth_pose(self, pose_list):
        poses_path = random.sample(pose_list, 1)[0]
        pcd_points = np.load(poses_path, allow_pickle=True)
        return pcd_points

    def get_default_config(self):
        particle_radius = self.cloth_particle_radius
        # cam_pos, cam_angle = np.array([-0.0, 0.65, 0.0]), np.array(
        cam_pos, cam_angle = np.array([-0.0, 1.5, 0.0]), np.array(
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
        self.render(mode="rgb_array")
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

    def get_corner_idxs(self):
        """Get the corner idxs of the cloth mesh"""
        x, y = self.config['ClothSize']
        corner_idxs = np.array([
            0, x-1, x*(y-1), x*y-1
        ])
        return corner_idxs

    def get_edge_idxs(self):
        """Get edge idxs of the cloth mesh"""
        x, y = self.config['ClothSize']
        edge_idxs = np.array([
            np.arange(x), np.arange(x*(y-1), x*y), np.arange(0, x*y, x), np.arange(x-1, x*y, x)
        ]).flatten()
        return edge_idxs

    def get_observations(self, cloth_only=True):
        rgb, depth = self._get_rgbd(cloth_only=cloth_only)
        rgb = rgb[::-1, :, :]  # reverse the height dimension
        depth = depth[::-1, :]  # reverse the height dimension
        object_pcd_points = self._get_cloud()
        obs = {
            "color": rgb, 
            "depth": depth, 
            "object_pcd_points": object_pcd_points,
            "goal_pcd_points": self.goal_pcd_points,
            "action_location_score": 0.,
            "poke_idx": 0,
            "cloth_corners": object_pcd_points[self.get_corner_idxs()],
        }
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

    def reset(self):
        self.set_scene()
        self.particle_num = pyflex.get_n_particles()
        self.prev_reward = 0.0
        self.time_step = 0
        if self.starts_list != []:
            reset_state = self._sample_cloth_pose(self.starts_list)
            reset_state = np.concatenate([reset_state, np.ones([reset_state.shape[0], 1])], axis=1)
            pyflex.set_positions(reset_state)
            pyflex.step()
        else:
            self._set_to_flat()
        if hasattr(self, "action_tool"):
            self.action_tool.reset([0, 0.1, 0])
            self._set_picker_pos(self.reset_pos)
        # if self.recording:
            # self.video_frames.append(self.render(mode="rgb_array"))

        if self.goals_list != []:
            self.goal_pcd_points = self._sample_cloth_pose(self.goals_list)

        obs = self.get_observations(cloth_only=False)

        return obs

    def step(
        self,
        action,
    ):
        """If record_continuous_video is set to True, will record an image for each sub-step"""
        if self.record: 
            self.start_record()
            self.video_frames.append(self.get_image())

        action_scaled = self._step(action)
        nobs = self.get_observations(cloth_only=False)

        self.time_step += 1

        if self.record:
            self.video_frames.append(self.get_image())
            info = {
                "cam_frames": copy.copy(self.video_frames)
            }
            del self.video_frames
        else: 
            info = {}
        info['action_scaled'] = action_scaled
        return nobs, info

    def _step(self, action):
        """Action is the 3D coordinate of a cloth node and the flow of the action"""
        location, action_flow = action

        # Teleport picker to location and grasp
        self._set_picker_pos(location)
        grasp_action = np.concatenate([location, [1]], axis=0)
        self.action_tool.step(grasp_action, render=not self.headless)

        if self.planar_action:
            action_scaled = np.zeros((3,))
            action_scaled[0] = action_flow[0]*self.max_action_scale[0] # x
            action_scaled[2] = action_flow[1]*self.max_action_scale[2] # y
            raise_z = np.array([0, 0.075, 0]) 
            # lower_z = np.array([0, -0.05, 0]) 

            pick = np.concatenate([location+raise_z, [1]], axis=0)
            move = np.concatenate([location+raise_z+action_scaled, [1]], axis=0)
            # lower = np.concatenate([location+raise_z+action_scaled+lower_z, [1]], axis=0)
            # drop = np.concatenate([location+raise_z+action_scaled+lower_z, [0]], axis=0)
            drop = np.concatenate([location+raise_z+action_scaled, [0]], axis=0)

            # sub_actions = [pick, move, lower, drop]
            sub_actions = [pick, move, drop]
            for sub in sub_actions:
                self.action_tool.step(sub, render=not self.headless)
                # for _ in range(5):
                #     pyflex.step()
        else:
            action_scaled = action_flow.copy()
            action_scaled[0] *= self.max_action_scale[0]
            action_scaled[1] = ((action_scaled[1] + 1) / 2) * self.max_action_scale[1] # rescaled [-1, 1] to [0, max]
            action_scaled[2] *= self.max_action_scale[2]
            # action_scaled = action_flow * 0.1

            move_action = np.concatenate([location + action_scaled, [1]], axis=0)
            self.action_tool.step(move_action, render=not self.headless)

            drop_action = np.concatenate([location + action_scaled, [0]], axis=0)
            self.action_tool.step(drop_action, render=not self.headless)

        # go to neutral position
        self._set_picker_pos(self.reset_pos)
        for _ in range(self.wait_steps):
            self.action_tool.step(self.reset_act, render=not self.headless)
        
        return action_scaled

    def unscale_action(self, action_param_envscaled):
        if self.planar_action:
            action_unscaled = action_param_envscaled.copy()
            action_unscaled[0] /= self.max_action_scale[0] # x 
            action_unscaled[1] /= self.max_action_scale[2] # y
        else:
            action_unscaled = action_param_envscaled.copy()
            action_unscaled[0] /= self.max_action_scale[0]
            action_unscaled[1] = (action_unscaled[1] / self.max_action_scale[1]) * 2 - 1 # rescaled [0, max] to [-1, 1]
            action_unscaled[2] /= self.max_action_scale[2]
        return action_unscaled

    def _get_info(self):
        return {}
