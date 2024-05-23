import numpy as np
import open3d as o3d
import scipy as sp
import h5py

if __name__ == "__main__":

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    # model = o3d.io.read_triangle_model("dataset/area_3/3d/rgb.obj")
    # mesh = o3d.geometry.TriangleMesh()

    # for m in model.meshes:
    #     mesh += m.mesh

    # mat = h5py.File("dataset/area_3/3d/pointcloud.mat")
    # roomptr = mat["Area_3"]["Disjoint_Space"]["object"][14][0]
    # pcd = o3d.geometry.PointCloud()

    # for obj in mat[roomptr]["points"]:
    #     pts = np.asarray(mat[obj[0]])

    #     pcd.points.extend(pts.T)

    # R = mesh.get_rotation_matrix_from_xyz((np.deg2rad(270), 0, 0))
    # pcd.rotate(R, center=(0, 0, 0))

    mesh = o3d.io.read_triangle_mesh("dataset/area_3/3d/rgb.obj")
    pcd = o3d.io.read_point_cloud("dataset/area_3/Area_3/office_9/office_9.txt", format="xyz")

    R = pcd.get_rotation_matrix_from_xyz((np.deg2rad(-90), 0, 0))
    pcd.rotate(R, center=(0, 0, 0))

    print(pcd)
    print(mesh)

    # mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    # print(mesh.cluster_connected_triangles()[0])

    # pcd = mesh.sample_points_poisson_disk(500)

    o3d.visualization.draw_geometries([mesh, pcd])
