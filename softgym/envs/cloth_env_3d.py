import numpy as np
import pyflex
import copy
import random
from pathlib import Path
from softgym.envs.cloth_env import FlexEnv
from softgym.action_space.action_space import PickerPickPlace
from softgym.utils.gemo_utils import *


def get_visible_idxs(particle_pos, depth, extrinsic_matrix, zthresh=0.01):
    # project particle_pos into depth image coordinates
    visible_idxs, hidden_idxs = [], []
    height, width = depth.shape
    K = intrinsic_from_fov(height, width, 45) # the fov is 90 degrees
    for i, pos in enumerate(particle_pos):

        # project the point into the image
        pos_h = np.array([pos[0], pos[1], pos[2], 1.0]) # x z y -> x z y 1 
        cam_coord = extrinsic_matrix.round(7) @ pos_h
        
        u0 = K[0, 2]
        v0 = K[1, 2]
        fx = K[0, 0]
        fy = K[1, 1]

        cam_coord = cam_coord.round(7)
        x, y, z = cam_coord[0], cam_coord[1], cam_coord[2]
        u = x * fx / z + u0
        v = y * fy / z + v0
        u = int(np.rint(u))
        v = int(np.rint(v))

        # Check if points are out of bounds
        if u < 0 or u >= width or v < 0 or v >= height:
            hidden_idxs.append(i)
            continue

        # depth at v, u can be 0 due to floating point inaccuracy
        # but this value would never be encountered in a real projection
        # as 0 corresponds to a non cloth pixel
        if depth[v, u] == 0:
            # check adjacent pixels
            # if they are nonzero, use the closest one
            # otherwise, check the next ring of pixels
            # if none of them are nonzero, then the point is not visible
            found = False
            for du in range(-5, 5):
                for dv in range(-5, 5):
                    if u + du < 0 or u + du >= width or v + dv < 0 or v + dv >= height:
                        continue
                    
                    if depth[v + dv, u + du] != 0:
                        u += du
                        v += dv
                        found = True
                        break
                if found:
                    break

        zdiff = np.abs(depth[v, u] - z)
        if zdiff < zthresh:
            visible_idxs.append(i)
        else:
            hidden_idxs.append(i)

    return visible_idxs, hidden_idxs

class ClothEnv3D(FlexEnv):
    def __init__(
        self,
        particle_radius=0.00625,
        picker_radius=0.005,
        num_pickers=1,
        success_threshold=0.003,
        headless=False,
        record=False,
        states_list=None,
        wait_steps=20,
        planar_action=True,
        max_action_scale = [0.125, 0.125, 0.125],
        use_subgoals=False,
        fold_unfold_ratio=0,
        cached_states_path=None,
        **kwargs
    ):
        self.cloth_particle_radius = particle_radius
        self.num_pickers = num_pickers
        self.success_threshold = success_threshold
        self.headless = headless
        self.record = record
        self.states_list = states_list
        self.wait_steps = wait_steps
        self.planar_action = planar_action
        self.max_action_scale = max_action_scale # x z y 
        self.use_subgoals = use_subgoals
        self.fold_unfold_ratio = fold_unfold_ratio
        super().__init__(headless=headless, **kwargs)

        if cached_states_path is not None and self.use_cached_states:
            self.get_cached_configs_and_states(cached_states_path, self.num_variations)

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
        self.extrinsic_matrix = get_extrinsic_matrix(self)
        self.corner_idxs = self.get_corner_idxs()
        self.edge_idxs = self.get_edge_idxs()

    def _sample_cloth_pose(self, pose_list):
        poses_path = random.sample(pose_list, 1)[0]
        pcd_points = np.load(poses_path, allow_pickle=True)
        return pcd_points, poses_path

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
        rgb = rgb[::-1, :, :]  # reverse the height dimension
        rgb = rgb[:, :, :3]
        depth = np.array(depth).reshape(
            self.camera_params["default_camera"]["height"],
            self.camera_params["default_camera"]["width"],
        )
        depth = depth[::-1, :]  # reverse the height dimension
        depth[depth >= 999] = 0  # use 0 instead of 999 for null
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
        rgb_cloth, depth_cloth = self._get_rgbd(cloth_only=True)
        rgb, depth = self._get_rgbd(cloth_only=cloth_only)
        object_pcd_points = self._get_cloud()
        visible_idxs, hidden_idxs = get_visible_idxs(object_pcd_points, depth_cloth, self.extrinsic_matrix)
        
        # make a plotly plot with the observable idxs
        if False:
            import plotly.graph_objects as go
            fig = go.Figure(data=[
                go.Scatter3d(
                x=object_pcd_points[observable_idxs, 0],
                y=object_pcd_points[observable_idxs, 2],
                z=object_pcd_points[observable_idxs, 1],
                mode='markers',
                marker=dict(
                    size=3,
                    color='red',
                    opacity=0.8
                    )
                ),
                go.Scatter3d(
                x=object_pcd_points[:, 0],
                y=object_pcd_points[:, 2],
                z=object_pcd_points[:, 1],
                mode='markers',
                marker=dict(
                    size=2,
                    color='blue',
                    opacity=0.8
                    )
                )
            ])
            fig.update_layout(scene_zaxis_range=[0, 0.5])
            fig.update_layout(scene_xaxis_range=[-0.3, 0.3])
            fig.update_layout(scene_yaxis_range=[-0.3, 0.3])
            fig.show()

        obs = {
            "color": rgb, 
            "depth": depth, 
            "object_pcd_points": object_pcd_points,
            "goal_pcd_points": self.goal_pcd_points,
            "action_location_score": 0.,
            "poke_idx": 0,
            # "cloth_corners": object_pcd_points[self.corner_idxs],
            # "cloth_edges": object_pcd_points[self.edge_idxs],
            "visible_idxs": visible_idxs,
            "hidden_idxs": hidden_idxs,
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

    def set_pyflex_positions(self, positions):
        if positions.shape[1] == 3: 
            positions = np.concatenate([positions, np.ones([positions.shape[0], 1])], axis=1)
        pyflex.set_positions(positions)
        pyflex.step()

    def reset(self, flip_cloth=False, start_state=None, goal_state=None):
        self.set_scene()
        self.particle_num = pyflex.get_n_particles()
        self.prev_reward = 0.0
        self.time_step = 0

        # if start state and goal state are both specified,
        # then load those states
        if goal_state is not None and start_state is not None:
            self.goal_pcd_points = np.load(goal_state, allow_pickle=True)
            self.set_pyflex_positions(np.load(start_state, allow_pickle=True))
        elif self.states_list is not None: # or, sample a start state and goal state
            prob = np.random.random()
            if prob < self.fold_unfold_ratio: # load folding goal
                self.goal_pcd_points, goal_path = self._sample_cloth_pose(self.states_list)
                if self.use_subgoals: # load start state as previous pose
                    reset_state = np.load(Path(goal_path).parent / 'prev_object_pcd.npy', allow_pickle=True)
                else:
                    reset_state = self._get_flat_pos()
                self.set_pyflex_positions(reset_state)
            else: # load unfolding goal
                self._set_to_flat()
                self.goal_pcd_points = pyflex.get_positions().reshape(-1, 4)[:, :3]
                reset_state, _ = self._sample_cloth_pose(self.states_list)

                if flip_cloth: 
                    self.goal_pcd_points = self._rotate_positions(self.goal_pcd_points)
                    self.set_pyflex_positions(reset_state)
                    reset_state = self._rotate_positions(reset_state)
                    self.set_pyflex_positions(reset_state)
                    max_z = reset_state[:, 1].max()
                    for i in range(int(max_z / 0.003) + 1):
                        pyflex.step()
                else:
                    self.set_pyflex_positions(reset_state)
        else: # otherwise, just set to flat
            self._set_to_flat()

        if hasattr(self, "action_tool"):
            self.action_tool.reset([0, 0.1, 0])
            self._set_picker_pos(self.reset_pos)
        # if self.recording:
            # self.video_frames.append(self.render(mode="rgb_array"))

        obs = self.get_observations(cloth_only=False)

        # plot start and goal states in plotly
        if False:
            import plotly.graph_objects as go
            fig = go.Figure(data=[
                go.Scatter3d(
                x=obs['object_pcd_points'][:, 0],
                y=obs['object_pcd_points'][:, 2],
                z=obs['object_pcd_points'][:, 1],
                mode='markers',
                marker=dict(
                    size=5,
                    color='blue',
                    opacity=0.8
                    )
                ),
                go.Scatter3d(
                x=obs['goal_pcd_points'][:, 0],
                y=obs['goal_pcd_points'][:, 2],
                z=obs['goal_pcd_points'][:, 1],
                mode='markers',
                marker=dict(
                    size=5,
                    color='red',
                    opacity=0.8
                    )
                )
            ])
            fig.update_layout(scene_zaxis_range=[0, 0.5])
            fig.update_layout(scene_xaxis_range=[-0.3, 0.3])
            fig.update_layout(scene_yaxis_range=[-0.3, 0.7])
            fig.show()

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

    def _rotate_positions(self, cloth_pos):
        max_z = cloth_pos[:, 1].max()
        cloth_pos_rot = cloth_pos[:, [0, 2, 1]] # x z y -> x y z

        # create a transformation matrix rotating 180 degrees about the y axis
        orig_to_rot_T = get_rotation_matrix(np.deg2rad(180), [0, 1, 0])
        orig_to_rot_T[:3, 3] = [0, 0, max_z+0.001]

        # transform cloth
        cloth_pos_rot = np.concatenate([cloth_pos_rot, np.ones([cloth_pos_rot.shape[0], 1])], axis=1)
        cloth_pos_rot = (orig_to_rot_T @ cloth_pos_rot.T).T
        cloth_pos_rot = cloth_pos_rot[:, [0, 2, 1]] # x y z -> x z y
        cloth_pos[:, :3] = cloth_pos_rot[:, :3]
        return cloth_pos
        
    def flip_cloth(self):
        """Flip the cloth mesh and let it settle before returning an observation"""
        # get cloth positions
        cloth_pos = pyflex.get_positions().reshape(-1, 4)
        cloth_pos_rot = self._rotate_positions(cloth_pos)

        # plotly 3D plot to show cloth_pos_rot
        if False:
            import plotly.graph_objects as go
            fig = go.Figure(data=[
                go.Scatter3d(
                x=cloth_pos_rot[:, 0],
                y=cloth_pos_rot[:, 1],
                z=cloth_pos_rot[:, 2],
                mode='markers',
                name="rotated",
                marker=dict(
                    size=5,
                    color='blue',
                    opacity=0.8
                    )
                ),
                go.Scatter3d(
                x=cloth_pos[:, 0],
                y=cloth_pos[:, 1],
                z=cloth_pos[:, 2],
                mode='markers',
                name="original",
                marker=dict(
                    size=5,
                    color='red',
                    opacity=0.8
                    )
                )
            ])
            fig.update_layout(scene_zaxis_range=[-0.2, 0.5])
            fig.update_layout(scene_xaxis_range=[-0.3, 0.3])
            fig.update_layout(scene_yaxis_range=[-0.3, 0.3])
            fig.show()

        self.set_pyflex_positions(cloth_pos)

        for i in range(100):
            pyflex.step()

    def _get_info(self):
        return {}
