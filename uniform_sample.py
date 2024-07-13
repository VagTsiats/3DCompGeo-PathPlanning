import open3d as o3d
import numpy as np
from tqdm import tqdm


def compute_triangle_areas(mesh):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    cross_product = np.cross(v1 - v0, v2 - v0)  # Compute the cross product of the edges of each triangle

    areas = np.linalg.norm(cross_product, axis=1) / 2  # Compute the area of each triangle

    return areas


def sample_point_in_triangle(v0, v1, v2):
    r1 = np.random.rand()
    r2 = np.random.rand()

    sqrt_r1 = np.sqrt(r1)
    u = 1 - sqrt_r1
    v = sqrt_r1 * (1 - r2)
    w = sqrt_r1 * r2

    sampled_point = u * v0 + v * v1 + w * v2

    return sampled_point


def sample_mesh_uniformly(mesh, num_samples):

    areas = compute_triangle_areas(mesh)

    cumulative_areas = np.cumsum(areas)  # Compute cumulative area for weighted sampling
    total_mesh_area = cumulative_areas[-1]

    # Generate random samples
    sampled_points = []
    for _ in tqdm(range(num_samples), desc="Sampling mesh"):
        # Randomly select a triangle based on the area
        r = np.random.rand() * total_mesh_area
        triangle_index = np.searchsorted(cumulative_areas, r)

        v0, v1, v2 = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)[triangle_index]]

        point = sample_point_in_triangle(v0, v1, v2)
        sampled_points.append(point)

    sampled_points_pcd = o3d.geometry.PointCloud()
    sampled_points_pcd.points = o3d.utility.Vector3dVector(np.vstack(sampled_points))

    return sampled_points_pcd


if __name__ == "__main__":
    mesh = o3d.io.read_triangle_mesh("dataset/area_1/RoomMesh.ply")
    sampled_points_pcd = sample_mesh_uniformly(mesh, (int)(1e7))
    o3d.io.write_point_cloud("dataset/area_1/RoomPointCloud_1e7.ply", sampled_points_pcd)
