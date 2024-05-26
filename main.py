import numpy as np
import open3d as o3d
import copy
import plane_detection_ransac


if __name__ == "__main__":

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    mesh = o3d.io.read_triangle_mesh("dataset/area_3/3d/semantic.obj")
    pcd = o3d.io.read_point_cloud("dataset/area_3/Area_3/office_9/office_9.txt", format="xyz")

    R = pcd.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))

    pcd.rotate(R, center=(0, 0, 0))

    bbox = pcd.get_minimal_oriented_bounding_box()
    bbox.scale(1.01, pcd.get_center())

    room_mesh = mesh.crop(bbox)

    print(room_mesh)

    room_mesh.compute_triangle_normals()

    room_pcd = o3d.geometry.PointCloud()
    room_pcd.points = o3d.utility.Vector3dVector(np.asarray(room_mesh.vertices))

    room_pcd = room_mesh.sample_points_uniformly(50000)

    o3d.visualization.draw_geometries([room_mesh, pcd])

    print(room_pcd)
    room_pcd.paint_uniform_color([0, 1, 0])

    # o3d.visualization.draw_geometries([room_mesh, room_pcd, bbox])

    planes_bbox = plane_detection_ransac.find_planes(room_pcd)

    # kai gamw doulevei

    # room_pcd.estimate_normals()
    # planes_bbox = room_pcd.detect_planar_patches(min_plane_edge_length=1, normal_variance_threshold_deg=20, outlier_ratio=0.2, coplanarity_deg=75)
    # planes_bbox.append(room_mesh)

    plane_points = set()
    for bb in planes_bbox:
        plane_cluster = bb.get_point_indices_within_bounding_box(room_pcd.points)
        plane_points.update(plane_cluster)

    pcd_no_planes = room_pcd.select_by_index(list(plane_points), invert=True)

    height_mask = np.asarray(pcd_no_planes.points)[:, 1] < bbox.get_center()[1] + 0.5

    pcd_no_planes_cut = o3d.geometry.PointCloud()
    pcd_no_planes_cut.points = o3d.utility.Vector3dVector(np.asarray(pcd_no_planes.points)[height_mask])
    pcd_no_planes_cut.paint_uniform_color([1, 0, 0])

    # height_mask = np.asarray(room_pcd.points)[:, 1] < bbox.get_center()[1] + 0.5

    # pcd_no_planes_cut = o3d.geometry.PointCloud()
    # pcd_no_planes_cut.points = o3d.utility.Vector3dVector(np.asarray(room_pcd.points)[height_mask])
    # pcd_no_planes_cut.paint_uniform_color([1, 0, 0])

    # χρησιμοποιώντας το extend των bboxes μπορώ να βρω με εξωτερικό γινόμενο το normal θεωρητικά και να βρώ ποιό είναι το πάτωμα

    geometries = planes_bbox
    geometries.append(pcd_no_planes_cut)
    geometries.append(room_mesh)

    o3d.visualization.draw_geometries(geometries)
