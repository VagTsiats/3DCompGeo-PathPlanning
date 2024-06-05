import open3d as o3d
import copy
import numpy as np
from tqdm import tqdm
from scipy.spatial import KDTree


def is_wall(normal, z_thresh=0.2):
    # Wall: Normal vector should be close to horizontal (i.e., near zero Z component)
    return abs(normal[1]) < z_thresh


def is_ceiling(normal, z_thresh=0.9, direction=1):
    # Ceiling: Normal vector should be close to vertical (i.e., Z component near 1)
    return normal[1] > z_thresh * direction


def is_floor(normal, z_thresh=0.9, direction=-1):
    # Floor: Normal vector should be close to vertical (i.e., Z component near -1)
    return normal[1] < z_thresh * direction


def do_voxels(mesh):

    room_mesh = copy.deepcopy(mesh)
    room_mesh.compute_vertex_normals()

    voxel_size = 0.05
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(room_mesh, voxel_size)

    # Extract voxel grid data
    voxels = voxel_grid.get_voxels()

    vertex_normals = np.asarray(room_mesh.vertex_normals)
    kdtree = KDTree(room_mesh.vertices)

    for voxel in tqdm(voxels):

        voxel_center = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)

        # distances = np.linalg.norm(np.asarray(mesh.vertices) - voxel_center, axis=1)
        # nearest_vertex_index = np.argmin(distances)
        # normal = np.asarray(mesh.vertex_normals)[nearest_vertex_index]

        _, idx = kdtree.query(voxel_center)
        normal = vertex_normals[idx]

        print(voxel)

        if is_wall(normal):
            voxel.color = [0.5, 0.5, 0.5]
        elif is_ceiling(normal):
            voxel.color = [0, 1, 0]
        elif is_floor(normal):
            voxel.color = [1, 0, 0]

        voxel_grid.add_voxel(voxel)

    # mymesh = room_mesh.crop(ceiling_grid.get_minimal_oriented_bounding_box())

    o3d.visualization.draw_geometries([voxel_grid])
