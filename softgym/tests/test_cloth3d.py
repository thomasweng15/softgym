import numpy as np
import pyflex
import time
from pathlib import Path

def load_cloth(path, scale=1.0):
    """Load .obj of cloth mesh. Only quad-mesh is acceptable!
    Return:
        - vertices: ndarray, (N, 3)
        - triangle_faces: ndarray, (S, 3)
        - stretch_edges: ndarray, (M1, 2)
        - bend_edges: ndarray, (M2, 2)
        - shear_edges: ndarray, (M3, 2)
    This function was written by Zhenjia Xu
    email: xuzhenjia [at] cs (dot) columbia (dot) edu
    website: https://www.zhenjiaxu.com/
    """
    # print("load cloth from: ", path)
    vertices, faces = [], []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        # 3D vertex
        if line.startswith('v '):
            vertices.append([float(n)
                             for n in line.replace('v ', '').split(' ')])
        # Face
        elif line.startswith('f '):
            idx = [n.split('/') for n in line.replace('f ', '').split(' ')]
            face = [int(n[0]) - 1 for n in idx]
            assert(len(face) == 4)
            faces.append(face)

    triangle_faces = []
    for face in faces:
        triangle_faces.append([face[0], face[1], face[2]])
        triangle_faces.append([face[0], face[2], face[3]])

    stretch_edges, shear_edges, bend_edges = set(), set(), set()

    # Stretch & Shear
    for face in faces:
        stretch_edges.add(tuple(sorted([face[0], face[1]])))
        stretch_edges.add(tuple(sorted([face[1], face[2]])))
        stretch_edges.add(tuple(sorted([face[2], face[3]])))
        stretch_edges.add(tuple(sorted([face[3], face[0]])))

        shear_edges.add(tuple(sorted([face[0], face[2]])))
        shear_edges.add(tuple(sorted([face[1], face[3]])))

    # Bend
    neighbours = dict()
    for vid in range(len(vertices)):
        neighbours[vid] = set()
    for edge in stretch_edges:
        neighbours[edge[0]].add(edge[1])
        neighbours[edge[1]].add(edge[0])
    for vid in range(len(vertices)):
        neighbour_list = list(neighbours[vid])
        N = len(neighbour_list)
        for i in range(N - 1):
            for j in range(i+1, N):
                bend_edge = tuple(
                    sorted([neighbour_list[i], neighbour_list[j]]))
                if bend_edge not in shear_edges:
                    bend_edges.add(bend_edge)

    return np.array(vertices) * scale, np.array(triangle_faces),\
        np.array(list(stretch_edges)), np.array(
            list(bend_edges)), np.array(list(shear_edges))
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
    dataset_path = Path("/data/stirumal/datasets/cloth3d/train/Jumpsuit")
    breakpoint()
    idx = "0001.obj"
    mesh_path = dataset_path / idx
    vertices, faces, stretch_edges, bend_edges, shear_edges = load_cloth(mesh_path, 1.0)
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
                              vertices.shape[0], 
                              faces.shape[0],
                              stretch_edges.shape[0],
                              bend_edges.shape[0],
                              shear_edges.shape[0],
                              *vertices.reshape(-1),
                              *faces.reshape(-1),
                              *stretch_edges.reshape(-1),
                              *bend_edges.reshape(-1),
                              *shear_edges.reshape(-1)])
    pyflex.init(headless, render, camera_width, camera_height)
    pyflex.set_scene(env_idx, scene_params, 0)
    x = 1
    while x < 1000:
        pyflex.step()
        pyflex.render()
        time.sleep(0.05)
        x += 1
    pass