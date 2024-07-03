import numpy as np
from collections import deque
from scipy.spatial import KDTree
from sklearn.datasets import make_blobs
import time
import open3d as o3d
from tqdm import tqdm


def dbscan_3d(setofpoints, eps, min_pts):
    n_points = setofpoints.shape[0]
    labels = -np.ones(n_points)
    noise_lbl = -1
    undefined_lbl = -1
    cluster_id = 0

    kdtree = KDTree(setofpoints)

    def expand_cluster(setofpoints, point_idx, clid, eps, min_pts):

        seeds = kdtree.query_ball_point(setofpoints[point_idx], r=eps)

        if len(seeds) < min_pts:
            labels[point_idx] = noise_lbl
            return False

        labels[seeds] = clid

        seeds_queue = deque(seeds)

        while seeds_queue:
            seed_idx = seeds_queue.popleft()
            result = kdtree.query_ball_point(setofpoints[seed_idx], r=eps)

            if len(result) >= min_pts:
                for res_idx in result:
                    if labels[res_idx] in [undefined_lbl, noise_lbl]:
                        if labels[res_idx] == undefined_lbl:
                            seeds_queue.append(res_idx)
                        labels[res_idx] = clid
        return True

    for point_idx in tqdm(range(n_points), desc="DBSCAN"):
        if labels[point_idx] == -1:
            if expand_cluster(setofpoints=setofpoints, point_idx=point_idx, clid=cluster_id, eps=eps, min_pts=min_pts):
                cluster_id += 1

    return labels.astype(int)


# Example usage
if __name__ == "__main__":
    # Generate sample 3D data (voxel grid)
    np.random.seed(42)
    voxel_grid = np.random.rand(int(1e2), 3)  # 100 points in 3D space

    centers = [[1, 1, 1], [-1, -1, -1], [1, -1, -1]]
    cluster_std = [0.3, 0.4, 0.2]
    n_samples = 100

    # Generate the data
    voxel_grid, _ = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=42)
    # print(voxel_grid)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(voxel_grid))

    # Run DBSCAN on voxel grid
    eps = 0.5  # Adjust eps for 3D space
    min_samples = 3
    labels = dbscan_3d(voxel_grid, eps, min_samples)
    print(labels)
    print(voxel_grid)
    print(voxel_grid[np.where(labels == -1)])

    labels1 = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_samples))

    print((labels == labels1).all())

    # Plot results
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(voxel_grid[:, 0], voxel_grid[:, 1], voxel_grid[:, 2], c=labels)
    ax.set_title("DBSCAN Clustering on 3D Voxel Grid")
    # plt.show()
