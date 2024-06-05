import open3d as o3d
import copy
import numpy as np
from tqdm import tqdm
from scipy.spatial import KDTree
from scipy.ndimage import label


class VoxelState:
    Empty = 0
    Ceiling = 1
    Floor = 2
    Wall = 3
    Occupied = 4


class VoxelSegment:

    def __init__(self, voxelgrid) -> None:
        self.color = [1, 0, 0]
        self.voxel_grid = voxelgrid

        self.min_bound = voxelgrid.get_min_bound()
        self.max_bound = voxelgrid.get_max_bound()
        self.voxel_size = voxelgrid.voxel_size
        self.dims = np.ceil((self.max_bound - self.min_bound) / self.voxel_size).astype(int)
        self.voxels = np.zeros(self.dims)

        for v in voxelgrid.get_voxels():
            self.voxels[v.grid_index[0], v.grid_index[1], v.grid_index[2]] = 1

    def add_voxel(self, idx):
        pass

    def get_voxel_center(self, idx):
        return self.min_bound + idx * self.voxel_size + [self.voxel_size / 2, self.voxel_size / 2, self.voxel_size / 2]

    def set_state(self, idx, state):
        self.voxels[idx[0], idx[1], idx[2]] = state

    def filter_normals(self):
        ceiling_mask = self.voxels == VoxelState.Ceiling
        ceiling_idx = np.nonzero(ceiling_mask)

        for i, v in enumerate(self.voxels[ceiling_mask]):
            self.voxels[ceiling_idx[0][i], ceiling_idx[1][i], ceiling_idx[2][i]] = self.filt_3d_26(
                [ceiling_idx[0][i], ceiling_idx[1][i], ceiling_idx[2][i]]
            )

        floor_mask = self.voxels == VoxelState.Floor
        floor_idx = np.nonzero(floor_mask)

        for i, v in enumerate(self.voxels[floor_mask]):
            self.voxels[floor_idx[0][i], floor_idx[1][i], floor_idx[2][i]] = self.filt_3d_26([floor_idx[0][i], floor_idx[1][i], floor_idx[2][i]])

    def ceiling_detection(self):
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                for k in range(self.dims[2]):
                    if k > 0:
                        if self.voxels[i, j, k] == VoxelState.Ceiling and self.voxels[i, j, k - 1] == VoxelState.Wall:
                            self.voxels[i, j, k] = VoxelState.Wall

                    # self.voxels[i, j, k] = self.filter_function([i, j, k], VoxelState.Ceiling)

        target_mask = self.voxels == VoxelState.Ceiling

        # scipy label uses 26 neighborhood
        labeled_grid, num_labels = label(self.voxels == VoxelState.Ceiling)

        # Iterate over each labeled segment
        for label_id in range(1, num_labels + 1):
            # Extract voxels belonging to current segment
            segment_mask = labeled_grid == label_id
            segment_voxels = target_mask * segment_mask.astype(int)

            occupied_volume = np.sum(segment_voxels)

            if occupied_volume < 300:
                self.voxels[segment_mask] = VoxelState.Occupied

        # fill holes
        # for i in tqdm(range(self.dims[0])):
        #     for j in range(self.dims[1]):
        #         for k in range(self.dims[2]):
        #             # fill holes
        #             if i > 0 and i < self.dims[0] - 1 and j > 0 and j < self.dims[1] - 1:
        #                 if (
        #                     self.voxels[i + 1, j, k] == VoxelState.Ceiling
        #                     or self.voxels[i - 1, j, k] == VoxelState.Ceiling
        #                     or self.voxels[i, j + 1, k] == VoxelState.Ceiling
        #                     or self.voxels[i, j - 1, k] == VoxelState.Ceiling
        #                 ):
        #                     self.voxels[i, j, k] = VoxelState.Ceiling

    def filt_3d_26(self, idx, state=None, threshold=3):
        values = np.zeros((5,))

        # Check 26-neighborhood
        num_neighbors = 0
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    if (
                        (di != 0 or dj != 0 or dk != 0)
                        and 0 <= idx[0] + di < self.voxels.shape[0]
                        and 0 <= idx[1] + dj < self.voxels.shape[1]
                        and 0 <= idx[2] + dk < self.voxels.shape[2]
                    ):
                        values[self.voxels[idx[0] + di, idx[1] + dj, idx[2] + dk].astype(int)] += 1
                        if self.voxels[idx[0] + di, idx[1] + dj, idx[2] + dk] == state:
                            num_neighbors += 1

        if state == None:
            values[0] = 0
            return np.argmax(values)

        if num_neighbors > threshold:
            return
        return 0

    def vis_state(self, state):
        pcd = o3d.geometry.PointCloud()
        voxel_centers = []
        voxel_colors = []

        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                for k in range(self.dims[2]):
                    if self.voxels[i, j, k] == state:
                        voxel_colors.append([1, 0, 0])
                        voxel_centers.append(self.get_voxel_center(np.array([i, j, k])))

        pcd.points = o3d.utility.Vector3dVector(np.array(voxel_centers))
        pcd.colors = o3d.utility.Vector3dVector(np.array(voxel_colors))

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=self.min_bound)

        o3d.visualization.draw_geometries([pcd, mesh_frame])

    def vis(self):
        pcd = o3d.geometry.PointCloud()
        voxel_centers = []
        voxel_colors = []

        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                for k in range(self.dims[2]):
                    if self.voxels[i, j, k] == VoxelState.Ceiling:
                        voxel_colors.append([1, 0, 0])
                        voxel_centers.append(self.get_voxel_center(np.array([i, j, k])))
                    elif self.voxels[i, j, k] == VoxelState.Floor:
                        voxel_colors.append([0, 1, 0])
                        voxel_centers.append(self.get_voxel_center(np.array([i, j, k])))
                    elif self.voxels[i, j, k] == VoxelState.Wall:
                        voxel_colors.append([0.5, 0.5, 0.5])
                        voxel_centers.append(self.get_voxel_center(np.array([i, j, k])))
                    elif self.voxels[i, j, k] == VoxelState.Occupied:
                        voxel_colors.append([0, 0, 0])
                        voxel_centers.append(self.get_voxel_center(np.array([i, j, k])))

        pcd.points = o3d.utility.Vector3dVector(np.array(voxel_centers))
        pcd.colors = o3d.utility.Vector3dVector(np.array(voxel_colors))

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=self.min_bound)

        o3d.visualization.draw_geometries([pcd, mesh_frame])


# class VoxelUtils:


def triangle_intersects_voxel(voxel_size, voxel_center, triangle_vertices):
    tri_min = np.min(triangle_vertices, axis=0)
    tri_max = np.max(triangle_vertices, axis=0)
    vox_min = voxel_center - voxel_size / 2
    vox_max = voxel_center + voxel_size / 2

    if np.all(vox_max >= tri_min) and np.all(vox_min <= tri_max):
        return True
    return False


def is_ceiling(normal, z_thresh=0.9, direction=-1):
    # Ceiling: Normal vector should be close to vertical (i.e., Z component near -1)
    return normal[2] < z_thresh * direction


def is_floor(normal, z_thresh=0.9, direction=1):
    # Floor: Normal vector should be close to vertical (i.e., Z component near 1)
    return normal[2] > z_thresh * direction


def do_voxels(mesh):

    room_mesh = copy.deepcopy(mesh)
    room_mesh.compute_vertex_normals()
    triangles = np.asarray(room_mesh.triangles)
    vertices = np.asarray(room_mesh.vertices)
    # vertex_normals = np.asarray(room_mesh.vertex_normals)
    mesh.compute_triangle_normals(normalized=True)
    triangle_normals = np.asarray(mesh.triangle_normals)
    triangle_centers = np.mean(vertices[triangles], axis=1)

    vertices_tree = KDTree(room_mesh.vertices)
    triangle_centers_tree = KDTree(triangle_centers)

    voxel_size = 0.05
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(room_mesh, voxel_size)
    voxels = voxel_grid.get_voxels()
    voxel_centers = np.array([voxel_grid.get_voxel_center_coordinate(v.grid_index) for v in voxels])
    voxel_coordinates = np.array([v.grid_index for v in voxels])
    voxel_colors = np.zeros_like(voxel_centers)

    ceiling = VoxelSegment(voxel_grid)

    seg = VoxelSegment(voxel_grid)

    # seg.vis()

    for i, voxel in enumerate(tqdm(voxels)):
        voxel_center = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)

        # find near triangles
        _, idx = triangle_centers_tree.query(voxel_center, 5)
        for tri_id in idx:
            triangle = triangles[tri_id]
            if triangle_intersects_voxel(voxel_size, voxel_center, vertices[triangle]):
                normal = triangle_normals[tri_id]

        if is_ceiling(normal):
            seg.set_state(voxel.grid_index, VoxelState.Ceiling)
        elif is_floor(normal):
            seg.set_state(voxel.grid_index, VoxelState.Floor)
        else:
            seg.set_state(voxel.grid_index, VoxelState.Wall)

    seg.vis()

    seg.filter_normals()

    seg.vis()

    seg.ceiling_detection()

    seg.vis()

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=voxel_grid.get_min_bound())

    # o3d.visualization.draw_geometries([voxel_grid, mesh_frame, room_mesh])


if __name__ == "__main__":

    room_mesh = o3d.io.read_triangle_mesh("dataset/area_1/RoomMesh.ply")

    do_voxels(room_mesh)
