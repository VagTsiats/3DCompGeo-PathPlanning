import open3d as o3d
import copy
import numpy as np
from tqdm import tqdm


def find_mesh_planes(t_mesh: o3d.geometry.TriangleMesh, threshold=0.5):

    mesh = copy.deepcopy(t_mesh)

    mesh.compute_triangle_normals(normalized=True)

    triangle_normals = np.asarray(mesh.triangle_normals)
    num_triangles = len(triangle_normals)
    triangle_labels = -np.ones(num_triangles, dtype=int)
    cluster_id = 0

    # label triangles based on face normal
    for tri1_id in tqdm(range(num_triangles)):
        if triangle_labels[tri1_id] != -1:
            continue

        triangle_labels[tri1_id] = cluster_id
        for tri2_id in range(tri1_id + 1, num_triangles):
            if np.linalg.norm(triangle_normals[tri1_id] - triangle_normals[tri2_id]) < threshold:
                triangle_labels[tri2_id] = cluster_id

        cluster_id += 1

    num_clusters = cluster_id

    triangles = np.asarray(mesh.triangles)

    adjacency_list = {i: set() for i in range(len(triangles))}

    for i, triangle in enumerate(triangles):
        for j in range(i + 1, len(triangles)):
            shared_vertices = set(triangle) & set(triangles[j])
            if len(shared_vertices) >= 2:
                adjacency_list[i].add(j)
                adjacency_list[j].add(i)

    # print(adjacency_list)

    # Assign cluster labels to triangles
    triangle_clusters = [[] for _ in range(num_clusters)]
    for tri_id, label in enumerate(triangle_labels):
        triangle_clusters[label].append(tri_id)

    # Visualize each cluster
    mesh.paint_uniform_color([1, 0, 0])
    for cluster in triangle_clusters:
        col = np.random.rand(3)
        if len(cluster) < 100:
            continue
        for tri_id in cluster:
            for vert_id in mesh.triangles[tri_id]:
                mesh.vertex_colors[vert_id] = col

    o3d.visualization.draw_geometries([mesh])
