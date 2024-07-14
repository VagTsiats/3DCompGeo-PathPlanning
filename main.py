import numpy as np
import open3d as o3d
import voxels


if __name__ == "__main__":

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    pcd = o3d.io.read_point_cloud("dataset/Stanford3dDataset_v1.2/Area_3/office_1/office_1.txt", format="xyz")
    pcd += o3d.io.read_point_cloud("dataset/Stanford3dDataset_v1.2/Area_3/office_2/office_2.txt", format="xyz")
    pcd += o3d.io.read_point_cloud("dataset/Stanford3dDataset_v1.2/Area_3/office_3/office_3.txt", format="xyz")
    pcd += o3d.io.read_point_cloud("dataset/Stanford3dDataset_v1.2/Area_3/office_4/office_4.txt", format="xyz")
    pcd += o3d.io.read_point_cloud("dataset/Stanford3dDataset_v1.2/Area_3/office_5/office_5.txt", format="xyz")
    pcd += o3d.io.read_point_cloud("dataset/Stanford3dDataset_v1.2/Area_3/office_6/office_6.txt", format="xyz")
    pcd += o3d.io.read_point_cloud("dataset/Stanford3dDataset_v1.2/Area_3/office_7/office_7.txt", format="xyz")
    pcd += o3d.io.read_point_cloud("dataset/Stanford3dDataset_v1.2/Area_3/office_8/office_8.txt", format="xyz")
    pcd += o3d.io.read_point_cloud("dataset/Stanford3dDataset_v1.2/Area_3/office_9/office_9.txt", format="xyz")
    pcd += o3d.io.read_point_cloud("dataset/Stanford3dDataset_v1.2/Area_3/office_10/office_10.txt", format="xyz")
    pcd += o3d.io.read_point_cloud("dataset/Stanford3dDataset_v1.2/Area_3/hallway_1/hallway_1.txt", format="xyz")
    pcd += o3d.io.read_point_cloud("dataset/Stanford3dDataset_v1.2/Area_3/hallway_2/hallway_2.txt", format="xyz")

    o3d.visualization.draw_geometries([pcd])

    # voxels.voxelize_mesh(room_mesh)
    voxels.voxelize_pcd(pcd)
