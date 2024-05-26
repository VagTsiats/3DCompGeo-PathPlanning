import open3d as o3d
import copy
import numpy as np


def find_planes(pcd: o3d.geometry.PointCloud):

    num_planes = 0

    pcd_no_planes = copy.deepcopy(pcd)

    bbox = pcd.get_axis_aligned_bounding_box()

    bbox_center = bbox.get_center()
    bbox_half_ext = bbox.get_half_extent()

    bbox_face_centroids = np.tile(bbox_center, (6, 1))

    for i in range(6):
        if i < 3:
            bbox_face_centroids[i, i] += bbox_half_ext[i]
        else:
            bbox_face_centroids[i, i - 3] -= bbox_half_ext[i - 3]

    bbox_cent = o3d.geometry.PointCloud()
    bbox_cent.points = o3d.utility.Vector3dVector(bbox_face_centroids)
    bbox_cent.paint_uniform_color([1, 0, 0])
    plane_bboxes = []

    while num_planes < 6:
        if len(pcd_no_planes.points) < 3:
            break
        plane_, inliers = pcd_no_planes.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=10000)

        # afou vrisko ta epipeda me strict oria na kanv fit ta shmeia toy point cloud pio xalara gia na vrisko ola ta shmeia toy epipedoy

        inlier_cloud = pcd_no_planes.select_by_index(inliers)

        plane_centroid = inlier_cloud.get_center()

        # distances_to_bbox_faces = [np.linalg.norm(plane_centroid - face_centroid) for face_centroid in bbox_face_centroids]
        distances_to_bbox_faces = distance_from_plane(bbox_face_centroids, plane_)

        idx = np.argmin(distances_to_bbox_faces)

        if distances_to_bbox_faces[idx] < 0.3:
            bbox_face_centroids = np.delete(bbox_face_centroids, idx, axis=0)
            inlier_cloud.paint_uniform_color([1, 0, 0])
            pl_bbx = inlier_cloud.get_oriented_bounding_box()
            pl_bbx.color = np.random.random((3, 1))
            plane_bboxes.append(pl_bbx)
            num_planes += 1
        else:
            inlier_cloud.paint_uniform_color([0, 0, 1])

        pcd_no_planes = pcd_no_planes.select_by_index(inliers, invert=True)
        o3d.visualization.draw_geometries([inlier_cloud, pcd_no_planes, bbox_cent])

    return plane_bboxes


def distance_from_plane(points, plane):
    """
    Calculate the distance of each point from the plane.

    :param points: NumPy array of shape (n, 3) where n is the number of points.
    :param plane_normal: A tuple or list of the plane normal vector coefficients (a, b, c).
    :param d: The d coefficient in the plane equation ax + by + cz + d = 0.
    :return: A NumPy array of distances of each point from the plane.
    """
    a, b, c, d = plane
    # Compute the numerator of the distance formula for each point
    numerator = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d)
    # Compute the denominator (same for all points)
    denominator = np.sqrt(a**2 + b**2 + c**2)
    # Compute the distance for each point
    distances = numerator / denominator
    return distances
