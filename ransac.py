import numpy as np


def fit_plane(points):
    """
    Fit a plane to a set of points.
    points: A 3x3 matrix where each row represents a 3D point.
    Returns:
        The plane's normal vector and a point on the plane.
    """
    # Ensure we have 3 points
    assert points.shape == (3, 3)

    # Compute the normal vector of the plane
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    normal = np.cross(v1, v2)

    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)

    # Return the normal and a point on the plane
    return normal, points[0]


def distance_from_plane(points, plane_normal, plane_point):
    """
    Compute the distance of a point from a plane.
    point: A 3D point.
    plane_normal: The normal vector of the plane.
    plane_point: A point on the plane.
    Returns:
        The perpendicular distance from the point to the plane.
    """
    return np.abs(np.dot(points - plane_point, plane_normal))


def ransac_voxel_grid(wall_mask, num_iterations=int(1e3), distance_threshold=1, orientation_threshold=0.1):
    best_plane_normal = None
    best_plane_point = None
    max_inliers = 0

    wall_grid = np.argwhere(wall_mask)

    for _ in range(num_iterations):
        # Randomly sample 3 points
        sample_points = wall_grid[np.random.choice(wall_grid.shape[0], 3, replace=False)]

        # Fit a plane to these points
        plane_normal, plane_point = fit_plane(sample_points)

        # Ensure the plane normal has the desired orientation
        if abs(plane_normal[2]) < orientation_threshold:

            # Find inliers
            distance = distance_from_plane(wall_grid, plane_normal, plane_point)
            inliers = wall_grid[distance < distance_threshold]

            # Update the best plane if this one has more inliers
            if len(inliers) > max_inliers:
                max_inliers = len(inliers)
                best_plane_normal = plane_normal
                best_plane_point = plane_point
                best_inliers = inliers

    return (best_plane_normal, best_plane_point), tuple(best_inliers.T)
