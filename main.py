import numpy as np
import open3d as o3d

if __name__ == "__main__":

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    pcd = o3d.io.read_triangle_model("dataset/living_room_obj_mtl/living-room.obj")
    # print(pcd.meshes[10].mesh)

    mesh = o3d.geometry.TriangleMesh()

    for m in pcd.meshes:
        # print(m)
        mesh += m.mesh

    print(mesh)

    # downpcd = pcd.voxel_down_sample(voxel_size=0.05)

    mesh.compute_triangle_normals()

    # print(np.asarray(mesh.triangle_normals))

    o3d.visualization.draw_geometries([mesh])
