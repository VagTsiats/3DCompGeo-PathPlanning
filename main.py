import numpy as np
import open3d as o3d
import copy
import plane_detection_ransac
import mesh_plane_w_normals
import voxels
from tqdm import tqdm
import uniform_sample


if __name__ == "__main__":

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    mesh = o3d.io.read_triangle_mesh("dataset/area_3/3d/semantic.obj")
    pcd = o3d.io.read_point_cloud("dataset/area_3/Area_3/office_9/office_9.txt", format="xyz")
    pcd += o3d.io.read_point_cloud("dataset/area_3/Area_3/office_10/office_10.txt", format="xyz")

    pcd += o3d.io.read_point_cloud("dataset/area_3/Area_3/office_1/office_1.txt", format="xyz")
    pcd += o3d.io.read_point_cloud("dataset/area_3/Area_3/office_2/office_2.txt", format="xyz")

    pcd += o3d.io.read_point_cloud("dataset/area_3/Area_3/office_5/office_5.txt", format="xyz")
    pcd += o3d.io.read_point_cloud("dataset/area_3/Area_3/office_6/office_6.txt", format="xyz")
    pcd += o3d.io.read_point_cloud("dataset/area_3/Area_3/office_7/office_7.txt", format="xyz")
    pcd += o3d.io.read_point_cloud("dataset/area_3/Area_3/office_8/office_8.txt", format="xyz")
    pcd += o3d.io.read_point_cloud("dataset/area_3/Area_3/hallway_1/hallway_1.txt", format="xyz")
    # pcd += o3d.io.read_point_cloud("dataset/area_3/Area_3/hallway_2/hallway_2.txt", format="xyz")

    R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))

    mesh.rotate(R, center=(0, 0, 0))

    mesh.compute_triangle_normals()

    # o3d.visualization.draw_geometries([mesh, pcd])

    bbox = pcd.get_minimal_oriented_bounding_box()
    bbox.scale(1.01, pcd.get_center())

    room_mesh = mesh.crop(bbox)

    print(room_mesh)

    room_mesh.compute_triangle_normals()

    room_pcd = o3d.geometry.PointCloud()
    room_pcd.points = o3d.utility.Vector3dVector(np.asarray(room_mesh.vertices))

    room_pcd = room_mesh.sample_points_uniformly(20000)

    # o3d.visualization.draw_geometries([room_mesh, pcd])

    o3d.visualization.draw_geometries([pcd])

    # voxels.voxelize_mesh(room_mesh)
    voxels.voxelize_pcd(pcd, 0.05)
