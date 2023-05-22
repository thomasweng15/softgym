import os
import numpy as np
import pyflex
import time
from pathlib import Path
import h5py

if __name__ == "__main__":

    particle_radius=0.00625
    cam_pos, cam_angle = np.array([-0.0, 1.5, 0.0]), np.array(
            [0, -np.pi / 2.0, 0.0]
        )
    render_mode = 2
    env3d = True
    envDrop = False 
    if env3d:
        config = {
                "ClothPos": [-0.15, 0.0, -0.15],
                "ClothSize": [int(0.30 / particle_radius), int(0.30 / particle_radius)],
                "ClothStiff": [
                    2.0,
                    0.5,
                    1.0,
                ],  # Stretch, Bend and Shear #0.8, 1., 0.9 #1.0, 1.3, 0.9
                "mass": 0.54,
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
    elif envDrop:
        config = {
            'ClothPos': [-1.6, 2.0, -0.8],
            'ClothSize': [64, 32],
            'ClothStiff': [0.9, 1.0, 0.9],  # Stretch, Bend and Shear
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': np.array([1.07199, 0.94942, 1.15691]),
                                   'angle': np.array([0.633549, -0.397932, 0]),
                                   'width': 720,
                                   'height': 720}},
            'flip_mesh': 0,
            'mass': 0.0054,
        }
    mass = config['mass']
    camera_params = config['camera_params'][config['camera_name']]
    headless = False
    camera_width = 720
    camera_height = 720
    render = True

    cfp = os.getenv('CLOTHFUNNELS_SLEEVE_PATH')
    idx = 0
    with h5py.File(cfp, 'r') as file:
        data_dict = file[list(file.keys())[idx]] 
        vertices = data_dict['mesh_verts'][:]
        faces = data_dict['mesh_faces'][:]
        stretch_edges = data_dict['mesh_stretch_edges'][:]
        bend_edges = data_dict['mesh_bend_edges'][:]
        shear_edges = data_dict['mesh_shear_edges'][:]
        positions = data_dict['particle_pos'][:]
        pass
    
    # breakpoint()
    env_idx = 1
    vertices = vertices.astype(np.float32)
    faces = faces.astype(np.float32)
    stretch_edges = stretch_edges.astype(np.float32)
    bend_edges = bend_edges.astype(np.float32)
    shear_edges = shear_edges.astype(np.float32)
    scene_params = np.array([*config['ClothPos'], 
                             *config['ClothSize'], 
                             *config['ClothStiff'], 
                              render_mode,
                             *camera_params['pos'][:], 
                             *camera_params['angle'][:], 
                              camera_params['width'], 
                              camera_params['height'], 
                              mass,
                              config['flip_mesh'], 
                              vertices.shape[0]/3, 
                              faces.shape[0]/3,
                              stretch_edges.shape[0]/2,
                              bend_edges.shape[0]/2,
                              shear_edges.shape[0]/2,
                              *vertices.reshape(-1),
                              *faces.reshape(-1),
                              *stretch_edges.reshape(-1),
                              *bend_edges.reshape(-1),
                              *shear_edges.reshape(-1)])
    pyflex.init(headless, render, camera_width, camera_height)
    pyflex.set_scene(env_idx, scene_params, 0)
    pyflex.set_positions(positions.astype(np.float32))
    x = 1
    while x < 1000:
        pyflex.step()
        pyflex.render()
        time.sleep(0.05)
        x += 1
    pass