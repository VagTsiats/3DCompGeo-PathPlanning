import open3d as o3d
import copy
import numpy as np
from tqdm import tqdm
from scipy.spatial import KDTree
import scipy.ndimage as nd
from enum import Enum
from matplotlib import pyplot as plt


class VoxelState:
    Empty = 0
    Occupied = 1
    Ceiling = 2
    CeilingWall = 3
    Floor = 4
    Wall = 5


class VoxelSegment:

    def __init__(self, voxelgrid) -> None:
        self.voxel_grid = voxelgrid

        self.min_bound = voxelgrid.get_min_bound()
        self.max_bound = voxelgrid.get_max_bound()
        self.voxel_size = voxelgrid.voxel_size
        self.dims = np.ceil((self.max_bound - self.min_bound) / self.voxel_size).astype(int)
        self.grid_state = np.zeros(self.dims, dtype=int)

        self.floor_height = 1000

        for v in voxelgrid.get_voxels():
            self.grid_state[v.grid_index[0], v.grid_index[1], v.grid_index[2]] = VoxelState.Occupied

    def get_voxel_center(self, idx):
        return self.min_bound + idx * self.voxel_size + [self.voxel_size / 2, self.voxel_size / 2, self.voxel_size / 2]

    def set_state(self, idx, state):
        self.grid_state[idx[0], idx[1], idx[2]] = state

    def filter_normals(self):
        ceiling_mask = self.grid_state == VoxelState.Ceiling
        ceiling_idx = np.nonzero(ceiling_mask)

        change_mask = np.zeros_like(self.grid_state)

        for i, v in enumerate(self.grid_state[ceiling_mask]):
            self.grid_state[ceiling_idx[0][i], ceiling_idx[1][i], ceiling_idx[2][i]] = self.filter_3d_26(
                [ceiling_idx[0][i], ceiling_idx[1][i], ceiling_idx[2][i]]
            )

        floor_mask = self.grid_state == VoxelState.Floor
        floor_idx = np.nonzero(floor_mask)

        for i, v in enumerate(self.grid_state[floor_mask]):
            self.grid_state[floor_idx[0][i], floor_idx[1][i], floor_idx[2][i]] = self.filter_3d_26(
                [floor_idx[0][i], floor_idx[1][i], floor_idx[2][i]]
            )

        wall_mask = self.grid_state == VoxelState.Wall
        wall_idx = np.nonzero(wall_mask)

        for i, v in enumerate(self.grid_state[wall_mask]):
            self.grid_state[wall_idx[0][i], wall_idx[1][i], wall_idx[2][i]] = self.filter_3d_26([wall_idx[0][i], wall_idx[1][i], wall_idx[2][i]])

        occ_mask = self.grid_state == VoxelState.Occupied
        occ_idx = np.nonzero(occ_mask)

        for i, v in enumerate(self.grid_state[occ_mask]):
            self.grid_state[occ_idx[0][i], occ_idx[1][i], occ_idx[2][i]] = self.filter_3d_26([occ_idx[0][i], occ_idx[1][i], occ_idx[2][i]])

    def ceiling_detection(self):
        for k in tqdm(range(self.dims[2])):
            for j in range(self.dims[1]):
                for i in range(self.dims[0]):
                    if k > 0:
                        if (self.grid_state[i, j, k] == VoxelState.Ceiling) and self.grid_state[i, j, k - 1] == VoxelState.Wall:
                            self.grid_state[i, j, k] = VoxelState.Wall

        target_mask = self.grid_state == VoxelState.Ceiling

        # scipy label uses 26 neighborhood to segment neighboring voxels
        labeled_grid, num_labels = nd.label(target_mask, nd.generate_binary_structure(3, 3))

        # discard ceiling segments with less than 300 voxels
        for label_id in tqdm(range(1, num_labels + 1)):
            segment_mask = labeled_grid == label_id
            segment_voxels = target_mask.astype(int) * segment_mask.astype(int)

            occupied_volume = np.sum(segment_voxels)

            if occupied_volume < 300:
                self.grid_state[segment_mask] = VoxelState.Occupied

    def floor_detection(self):
        # for k in range(self.dims[2] - 2, 0, -1):
        #     for j in range(self.dims[1]):
        #         for i in range(self.dims[0]):
        #             if k > 0:
        #                 if (self.grid_state[i, j, k] == VoxelState.Wall) and self.grid_state[i, j, k - 1] == VoxelState.Wall:
        #                     self.grid_state[i, j, k] = VoxelState.Wall

        target_mask = self.grid_state == VoxelState.Floor 
        
        # scipy label uses 26 neighborhood to segment neighboring voxels
        labeled_grid, num_labels = nd.label(target_mask, nd.generate_binary_structure(3, 3))

        # compute the height of each segment and keep the segment with the min height in the grid
        for label_id in tqdm(range(1, num_labels + 1)):
            segment_mask = labeled_grid == label_id
            idx = np.nonzero(segment_mask)

            mean_height = np.mean(idx[2])

            if np.sum(segment_mask)>500 and mean_height < self.floor_height:
                self.floor_height = mean_height

            if mean_height > 5:
                self.grid_state[segment_mask] = VoxelState.Occupied

    def wall_detection(self):
        target_mask = self.grid_state == VoxelState.Wall

        # scipy label uses 26 neighborhood to segment neighboring voxels
        labeled_grid, num_labels = nd.label(target_mask, nd.generate_binary_structure(3, 3))

        # compute the height of each segment and keep the segment with the min height in the grid
        for label_id in tqdm(range(1, num_labels + 1)):
            segment_mask = labeled_grid == label_id

            if np.sum(segment_mask) < 500:
                self.grid_state[segment_mask] = VoxelState.Occupied

    # dont use it
    def ceiling_refinements(self):
        ceiling_grid = np.zeros((self.dims[0], self.dims[1]), dtype=int)
        ceiling_mask = np.logical_or(self.grid_state == VoxelState.Ceiling, self.grid_state == VoxelState.CeilingWall)
        ceiling_idx = np.nonzero(ceiling_mask)

        # 2d ceiling grid creation
        for i in range(len(ceiling_idx[0])):
            if ceiling_grid[ceiling_idx[0][i], ceiling_idx[1][i]] != 0 and ceiling_grid[ceiling_idx[0][i], ceiling_idx[1][i]] > ceiling_idx[2][i]:
                ceiling_grid[ceiling_idx[0][i], ceiling_idx[1][i]] = ceiling_idx[2][i]
            elif ceiling_grid[ceiling_idx[0][i], ceiling_idx[1][i]] == 0:
                ceiling_grid[ceiling_idx[0][i], ceiling_idx[1][i]] = ceiling_idx[2][i]

        # ceiling hole detection
        holes_mask = ceiling_grid == 0
        ceiling_mask = ceiling_grid != 0

        labeled_grid, num_labels = nd.label(holes_mask, nd.generate_binary_structure(2, 2))

        for label_id in tqdm(range(1, num_labels + 1)):
            segment_mask = labeled_grid == label_id

            ceiling_idx = np.nonzero(segment_mask)

            # for i, v in enumerate([segment_mask]):
            #     self.grid_state[ceiling_idx[0][i], ceiling_idx[1][i], ceiling_idx[2][i]] = self.filter_2d_8(
            #         [ceiling_idx[0][i], ceiling_idx[1][i], ceiling_idx[2][i]]
            #     )

            # self.grid_state[segment_mask] = VoxelState.Ceiling

        plt.imshow(ceiling_grid)
        plt.show()


    def floor_refinements(self):
        zero = -100
        floor_grid = np.ones((self.dims[0], self.dims[1]), dtype=int)*zero
        floor_mask = np.logical_or(self.grid_state == VoxelState.Floor, self.grid_state == VoxelState.Ceiling)
        # floor_mask = np.logical_or(floor_mask, self.grid_state == VoxelState.Ceiling)
        floor_idx = np.nonzero(floor_mask)
        
        # 2d floor grid creation
        for i in range(len(floor_idx[0])):
            if floor_grid[floor_idx[0][i], floor_idx[1][i]] != zero and floor_grid[floor_idx[0][i], floor_idx[1][i]] > floor_idx[2][i]:
                floor_grid[floor_idx[0][i], floor_idx[1][i]] = floor_idx[2][i]
            elif floor_grid[floor_idx[0][i], floor_idx[1][i]] == zero:
                floor_grid[floor_idx[0][i], floor_idx[1][i]] = floor_idx[2][i]


        #fill empty floor that has occupied over it 
        ceiling_2d_mask = np.any(self.grid_state == VoxelState.Ceiling, axis=2)
        floor_2d_mask = np.any(self.grid_state == VoxelState.Floor, axis=2)

        fill_mask = np.logical_and(ceiling_2d_mask, 1 - floor_2d_mask)

        self.grid_state[:,:,np.round(self.floor_height).astype(int)] += fill_mask * VoxelState.Floor


        #mask to fill occupied voxels on floor level
        floor_occupied_2d_mask = self.grid_state[:,:,np.round(self.floor_height).astype(int)] == VoxelState.Occupied
        self.grid_state[:,:,np.round(self.floor_height).astype(int)][floor_occupied_2d_mask] = VoxelState.Floor



        plt.imshow(floor_occupied_2d_mask)
        plt.show()

    def fill_walls(self):
        for i in tqdm(range(self.dims[0])):
            for j in range(self.dims[1]):
                for k in range(self.dims[2] - 1, 1, -1):
                    if self.grid_state[i, j, k] == VoxelState.Wall:
                        if self.grid_state[i, j, k - 1] == VoxelState.Occupied:
                            self.grid_state[i, j, k - 1] = VoxelState.Wall

                        if k < self.dims[2] - 1 and self.grid_state[i, j, k + 1] == VoxelState.Wall:
                            if j > 0 and self.grid_state[i, j - 1, k] == VoxelState.Occupied:
                                self.grid_state[i, j - 1, k] = VoxelState.Wall
                            if j < self.dims[1] - 1 and self.grid_state[i, j + 1, k] == VoxelState.Occupied:
                                self.grid_state[i, j + 1, k] = VoxelState.Wall

                            if i > 0 and self.grid_state[i - 1, j, k] == VoxelState.Occupied:
                                self.grid_state[i - 1, j, k] = VoxelState.Wall
                            if i < self.dims[0] - 1 and self.grid_state[i + 1, j, k] == VoxelState.Occupied:
                                self.grid_state[i + 1, j, k] = VoxelState.Wall

    def filter_2d_8(self, grid, idx, state=None, threshold=1):
        values = {}

        num_neighbors = 0
        for di in range(-1, 2):
            for dj in range(-1, 2):
                if (di != 0 or dj != 0) and 0 <= idx[0] + di < grid.shape[0] and 0 <= idx[1] + dj < grid.shape[1]:
                    if not values[grid[idx[0] + di, idx[1] + dj]]:
                        values[grid[idx[0] + di, idx[1] + dj]] = 1
                    else:
                        values[grid[idx[0] + di, idx[1] + dj]] += 1

                    if grid[idx[0] + di, idx[1] + dj] == state:
                        num_neighbors += 1

        if state == None:
            return values[np.argmax(values)]

    def filter_3d_26(self, idx, state=None, threshold=3):
        values = np.zeros((10,))

        num_neighbors = 0
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    if (
                        (di != 0 or dj != 0 or dk != 0)
                        and 0 <= idx[0] + di < self.grid_state.shape[0]
                        and 0 <= idx[1] + dj < self.grid_state.shape[1]
                        and 0 <= idx[2] + dk < self.grid_state.shape[2]
                    ):
                        values[self.grid_state[idx[0] + di, idx[1] + dj, idx[2] + dk].astype(int)] += 1
                        if self.grid_state[idx[0] + di, idx[1] + dj, idx[2] + dk] == state:
                            num_neighbors += 1

        if state == None:
            values[0] = 0
            return np.argmax(values)

        if num_neighbors > threshold:
            return
        return 0

    def extract_input(self, state):
        "function to extract input inliers of the given state"
        pass

    def vis_state(self, state):
        pcd = o3d.geometry.PointCloud()
        voxel_centers = []
        voxel_colors = []

        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                for k in range(self.dims[2]):
                    if (self.grid_state[i, j, k] in state):
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
                    if self.grid_state[i, j, k] == VoxelState.Ceiling:
                        voxel_colors.append([1, 0, 0])
                        voxel_centers.append(self.get_voxel_center(np.array([i, j, k])))
                    elif self.grid_state[i, j, k] == VoxelState.Floor:
                        voxel_colors.append([0, 1, 0])
                        voxel_centers.append(self.get_voxel_center(np.array([i, j, k])))
                    elif self.grid_state[i, j, k] == VoxelState.Wall:
                        voxel_colors.append([0.6, 0.6, 0.6])
                        voxel_centers.append(self.get_voxel_center(np.array([i, j, k])))
                    elif self.grid_state[i, j, k] == VoxelState.CeilingWall:
                        voxel_colors.append([0, 0, 1])
                        voxel_centers.append(self.get_voxel_center(np.array([i, j, k])))
                    elif self.grid_state[i, j, k] == VoxelState.Occupied:
                        voxel_colors.append([0.1, 0.1, 0.1])
                        voxel_centers.append(self.get_voxel_center(np.array([i, j, k])))

        pcd.points = o3d.utility.Vector3dVector(np.array(voxel_centers))
        pcd.colors = o3d.utility.Vector3dVector(np.array(voxel_colors))

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=self.min_bound)

        # o3d.visualization.draw_geometries([pcd, mesh_frame])

        return pcd


def triangle_intersects_voxel(voxel_size, voxel_center, triangle_vertices):
    tri_min = np.min(triangle_vertices, axis=0)
    tri_max = np.max(triangle_vertices, axis=0)
    vox_min = voxel_center - voxel_size / 2
    vox_max = voxel_center + voxel_size / 2

    if np.all(vox_max >= tri_min) and np.all(vox_min <= tri_max):
        return True
    return False


def is_ceiling(normal, z_thresh=0.9, direction=-1):
    return normal[2] < z_thresh * direction


def is_floor(normal, z_thresh=0.9, direction=1):
    return normal[2] > z_thresh * direction


def is_wall(normal, z_thresh=0.2):
    return normal[2] < z_thresh


def do_voxels(mesh):

    room_mesh = copy.deepcopy(mesh)
    triangles = np.asarray(room_mesh.triangles)
    vertices = np.asarray(room_mesh.vertices)
    room_mesh.compute_triangle_normals(normalized=True)
    triangle_normals = np.asarray(room_mesh.triangle_normals)
    triangle_centers = np.mean(vertices[triangles], axis=1)

    triangle_centers_tree = KDTree(triangle_centers)

    voxel_size = 0.05
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(room_mesh, voxel_size)
    voxels = voxel_grid.get_voxels()

    room_mesh.compute_vertex_normals()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=voxel_grid.get_min_bound())
    # o3d.visualization.draw_geometries([mesh_frame, room_mesh])

    seg = VoxelSegment(voxel_grid)

    for voxel in tqdm(voxels):
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
        elif is_wall(normal):
            seg.set_state(voxel.grid_index, VoxelState.Wall)
        else:
            # seg.set_state(voxel.grid_index, VoxelState.Wall)
            seg.set_state(voxel.grid_index, VoxelState.Occupied)

    

def reconstruction(voxelgrid:VoxelSegment):

    # pcd = seg.vis()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=voxelgrid.voxel_grid.get_min_bound())
    o3d.visualization.draw_geometries([mesh_frame, room_mesh, pcd])


    voxelgrid.filter_normals()

    voxelgrid.ceiling_detection()
    # seg.ceiling_refinements()  # more of a reconstruction

    voxelgrid.floor_detection()
    voxelgrid.floor_refinements()
    voxelgrid.vis_state([VoxelState.Floor])

    voxelgrid.wall_detection()
    voxelgrid.fill_walls()

    pcd = voxelgrid.vis()
    o3d.visualization.draw_geometries([mesh_frame, pcd])

    pass


if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    room_mesh = o3d.io.read_triangle_mesh("dataset/area_1/RoomMesh.ply")

    do_voxels(room_mesh)
