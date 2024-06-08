import open3d as o3d
import copy
import numpy as np
from tqdm import tqdm
from scipy.spatial import KDTree
import scipy.ndimage as nd
from enum import Enum
from matplotlib import pyplot as plt
import uniform_sample


class VoxelState:
    Empty = 0
    Occupied = 1
    Ceiling = 2
    Floor = 3
    Wall = 4


class VoxelGrid:

    def __init__(self, voxelgrid) -> None:
        self.voxel_grid = voxelgrid

        self.min_bound = voxelgrid.get_min_bound()
        self.max_bound = voxelgrid.get_max_bound()
        self.voxel_size = voxelgrid.voxel_size
        self.dims = np.ceil((self.max_bound - self.min_bound) / self.voxel_size).astype(int)

        self.grid_state = np.zeros(self.dims, dtype=int)
        self.grid_normals = np.zeros(self.dims, dtype=int)

        self.floor_height = 1000

        for v in voxelgrid.get_voxels():
            self.grid_state[v.grid_index[0], v.grid_index[1], v.grid_index[2]] = VoxelState.Occupied

    def get_voxel_center(self, idx):
        return self.min_bound + idx * self.voxel_size + [self.voxel_size / 2, self.voxel_size / 2, self.voxel_size / 2]

    def set_state(self, idx, state):
        self.grid_state[idx[0], idx[1], idx[2]] = state

    # def set_normals(self, idx, normal):
    #     self.grid_normals[idx[0], idx[1], idx[2]] = normal

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

    # def ceiling_detection(self):
    #     # Pass walls through Ceiling
    #     for k in range(self.dims[2]):
    #         for j in range(self.dims[1]):
    #             for i in range(self.dims[0]):
    #                 if k > 0:
    #                     if (self.grid_state[i, j, k] == VoxelState.Ceiling) and self.grid_state[i, j, k - 1] == VoxelState.Wall:
    #                         self.grid_state[i, j, k] = VoxelState.Wall

    #     # Discard small Ceiling segments
    #     target_mask = self.grid_state == VoxelState.Ceiling

    #     # scipy label uses 26 neighborhood to segment neighboring voxels
    #     labeled_grid, num_labels = nd.label(target_mask, nd.generate_binary_structure(3, 3))

    #     for label_id in range(1, num_labels + 1):
    #         segment_mask = labeled_grid == label_id
    #         segment_voxels = target_mask.astype(int) * segment_mask.astype(int)

    #         occupied_volume = np.sum(segment_voxels) * self.voxel_size

    #         if occupied_volume < 20:
    #             self.grid_state[segment_mask] = VoxelState.Occupied

    #     # No occupied on ceiling level
    #     for k in range(self.dims[2]):
    #         for j in range(self.dims[1]):
    #             for i in range(self.dims[0]):
    #                 if self.grid_state[i, j, k] == VoxelState.Occupied:
    #                     if self.filter_2d_8(grid=self.grid_state[:, :, k], idx=[i, j], state=VoxelState.Ceiling, d=5):
    #                         self.grid_state[i, j, k] = VoxelState.Ceiling

    def ceiling_floor_detection(self):
        # Pass walls through Ceiling
        for k in range(self.dims[2]):
            for j in range(self.dims[1]):
                for i in range(self.dims[0]):
                    if k > 0:
                        if (self.grid_state[i, j, k] == VoxelState.Ceiling) and self.grid_state[i, j, k - 1] == VoxelState.Wall:
                            self.grid_state[i, j, k] = VoxelState.Wall

        # Discard small Ceiling segments
        target_mask = self.grid_state == VoxelState.Ceiling

        labeled_grid, num_labels = nd.label(
            target_mask, nd.generate_binary_structure(3, 3)
        )  # scipy label uses 26 neighborhood to segment neighboring voxels

        for label_id in tqdm(range(1, num_labels + 1), "Ceiling & Floor Detection"):
            segment_mask = labeled_grid == label_id
            idx = np.nonzero(segment_mask)
            segment_voxels = target_mask.astype(int) * segment_mask.astype(int)

            occupied_volume = np.sum(segment_mask) * self.voxel_size

            if occupied_volume < 17:
                self.grid_state[segment_mask] = VoxelState.Occupied
                continue

            mean_height = np.mean(idx[2])

            if mean_height < self.dims[2] / 2:
                if mean_height < 5:
                    self.grid_state[segment_mask] = VoxelState.Floor
                    self.floor_height = (self.floor_height + mean_height) / 2
                else:
                    self.grid_state[segment_mask] = VoxelState.Occupied

        # No occupied on ceiling level
        for k in range(self.dims[2]):
            for j in range(self.dims[1]):
                for i in range(self.dims[0]):
                    if self.grid_state[i, j, k] == VoxelState.Occupied:
                        if self.filter_2d_8(grid=self.grid_state[:, :, k], idx=[i, j], state=VoxelState.Ceiling, d=5):
                            self.grid_state[i, j, k] = VoxelState.Ceiling

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

            if np.sum(segment_mask) * self.voxel_size > 25 and mean_height < self.floor_height:
                self.floor_height = mean_height

            if mean_height > 5:
                self.grid_state[segment_mask] = VoxelState.Occupied

    def wall_detection(self):
        target_mask = self.grid_state == VoxelState.Wall

        # scipy label uses 26 neighborhood to segment neighboring voxels
        labeled_grid, num_labels = nd.label(target_mask, nd.generate_binary_structure(3, 3))

        # compute the height of each segment and keep the segment with the min height in the grid
        for label_id in tqdm(range(1, num_labels + 1), desc="Wall detection"):
            segment_mask = labeled_grid == label_id

            if np.sum(segment_mask) < 500:
                self.grid_state[segment_mask] = VoxelState.Occupied

    def floor_refinements(self):
        zero = -100
        floor_grid = np.ones((self.dims[0], self.dims[1]), dtype=int) * zero
        floor_mask = np.logical_or(self.grid_state == VoxelState.Floor, self.grid_state == VoxelState.Ceiling)
        # floor_mask = np.logical_or(floor_mask, self.grid_state == VoxelState.Ceiling)
        floor_idx = np.nonzero(floor_mask)

        # 2d floor grid creation
        for i in range(len(floor_idx[0])):
            if floor_grid[floor_idx[0][i], floor_idx[1][i]] != zero and floor_grid[floor_idx[0][i], floor_idx[1][i]] > floor_idx[2][i]:
                floor_grid[floor_idx[0][i], floor_idx[1][i]] = floor_idx[2][i]
            elif floor_grid[floor_idx[0][i], floor_idx[1][i]] == zero:
                floor_grid[floor_idx[0][i], floor_idx[1][i]] = floor_idx[2][i]

        # fill empty floor that has occupied over it
        ceiling_2d_mask = np.any(self.grid_state == VoxelState.Ceiling, axis=2)
        floor_2d_mask = np.any(self.grid_state == VoxelState.Floor, axis=2)

        fill_mask = np.logical_or(ceiling_2d_mask, floor_2d_mask)

        self.grid_state[:, :, np.round(self.floor_height).astype(int)][fill_mask] = VoxelState.Floor

        # mask to fill occupied voxels on floor level
        floor_occupied_2d_mask = self.grid_state[:, :, np.round(self.floor_height).astype(int)] == VoxelState.Occupied
        self.grid_state[:, :, np.round(self.floor_height).astype(int)][floor_occupied_2d_mask] = VoxelState.Floor

        # plt.imshow(fill_mask)
        # plt.show()

    # def occupied_detection(self):
    #     ceiling_2d_mask = np.any(self.grid_state == VoxelState.Ceiling, axis=2)

    #     for k in tqdm(range(self.dims[2] - 1, 1, -1), desc="Occupied Detection"):
    #         for i in range(self.dims[0]):
    #             for j in range(self.dims[1]):
    #                 if self.grid_state[i, j, k] == VoxelState.Wall and ceiling_2d_mask[i, j]:
    #                     # if  self.grid_state[i,j,k-1] != VoxelState.Floor:
    #                     self.grid_state[i, j, k] = VoxelState.Occupied

    def fill_walls(self):
        # change_mask = np.zeros_like(self.grid_state)

        for i in tqdm(range(self.dims[0]), "Wall Treatment"):
            for j in range(self.dims[1]):
                for k in range(self.dims[2] - 1, 1, -1):
                    if self.grid_state[i, j, k] == VoxelState.Wall:

                        if self.grid_state[i, j, k - 1] == VoxelState.Occupied:
                            self.grid_state[i, j, k - 1] = VoxelState.Wall

                        if self.grid_state[i, j, k - 1] == VoxelState.Floor:
                            self.grid_state[i, j, k - 1] = VoxelState.Wall

                        # apo panw toixos kai gyrw occupied -> wall
                        # if k < self.dims[2] - 1 and self.grid_state[i, j, k + 1] == VoxelState.Wall:
                        #     if j > 0 and self.grid_state[i, j - 1, k] == VoxelState.Occupied:
                        #         change_mask[i, j - 1, k] = 1
                        #     if j < self.dims[1] - 1 and self.grid_state[i, j + 1, k] == VoxelState.Occupied:
                        #         change_mask[i, j + 1, k] = 1

                        #     if i > 0 and self.grid_state[i - 1, j, k] == VoxelState.Occupied:
                        #         change_mask[i - 1, j, k] = 1
                        #     if i < self.dims[0] - 1 and self.grid_state[i + 1, j, k] == VoxelState.Occupied:
                        #         change_mask[i + 1, j, k] = 1

        # self.grid_state[change_mask] = VoxelState.Wall

    def filter_2d_8(self, grid, idx, state=None, threshold=1, d=1):
        states_instances_dict = np.zeros((10,))

        num_neighbors = 0
        for di in range(-d, d + 1):
            for dj in range(-d, d + 1):
                if (di != 0 or dj != 0) and 0 <= idx[0] + di < grid.shape[0] and 0 <= idx[1] + dj < grid.shape[1]:
                    states_instances_dict[grid[idx[0] + di, idx[1] + dj]] += 1

                    if grid[idx[0] + di, idx[1] + dj] == state:
                        num_neighbors += 1

        if state == None:
            return states_instances_dict[np.argmax(states_instances_dict)]

        if num_neighbors > threshold:
            return True

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

    def triangle_intersects_voxel(self, idx, triangle_vertices):
        tri_min = np.min(triangle_vertices, axis=0)
        tri_max = np.max(triangle_vertices, axis=0)
        vox_min = self.voxel_grid.get_voxel_center_coordinate(idx) - self.voxel_size / 2
        vox_max = self.voxel_grid.get_voxel_center_coordinate(idx) + self.voxel_size / 2

        if np.all(vox_max >= tri_min) and np.all(vox_min <= tri_max):
            return True
        return False

    def extract_inliers_mesh(self, input: o3d.geometry.TriangleMesh, state):
        "function to extract input inliers of the given state"

        def is_point_in_voxel_grid(point):
            voxel = self.voxel_grid.get_voxel(point)
            try:
                return self.grid_state[voxel[0], voxel[1], voxel[2]] == state
            except:
                return False

        triangle_inliers = []

        triangles = np.asarray(input.triangles)
        vertices = np.asarray(input.vertices)

        for triangle in tqdm(triangles, desc="Inlier Extraction"):
            v0 = vertices[triangle[0]]
            v1 = vertices[triangle[1]]
            v2 = vertices[triangle[2]]

            # Check if any vertex of the triangle is within the voxel grid
            in_voxel_count = 0
            if is_point_in_voxel_grid(v0):
                in_voxel_count += 1
            if is_point_in_voxel_grid(v1):
                in_voxel_count += 1
            if is_point_in_voxel_grid(v2):
                in_voxel_count += 1

            # If at least two vertices are within the voxel grid, consider this triangle as an inlier
            if in_voxel_count >= 2:
                triangle_inliers.append(triangle)

        triangle_inliers = np.array(triangle_inliers)

        inlier_mesh = o3d.geometry.TriangleMesh()
        inlier_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        inlier_mesh.triangles = o3d.utility.Vector3iVector(triangle_inliers)
        inlier_mesh.compute_vertex_normals()

        o3d.visualization.draw_geometries([inlier_mesh])

        pass

    def extract_inliers_pcd(self, input: o3d.geometry.PointCloud, state):

        def is_point_in_voxel_grid(point):
            voxel = self.voxel_grid.get_voxel(point)
            try:
                return self.grid_state[voxel[0], voxel[1], voxel[2]] == state
            except:
                return False

        point_inliers = []

        points = np.asarray(input.points)

        for p in tqdm(points, desc="Inlier Extraction"):
            if is_point_in_voxel_grid(p):
                point_inliers.append(p)

        # Optionally, create a new mesh with the inlier triangles
        inlier_pcd = o3d.geometry.PointCloud()
        inlier_pcd.points = o3d.utility.Vector3dVector(np.array(point_inliers))

        # Visualize the inlier mesh
        o3d.visualization.draw_geometries([inlier_pcd])

    def visualize_grid_state(self, states):
        "visualizes voxel centers that belong to the given states list"
        pcd = o3d.geometry.PointCloud()
        voxel_centers = []
        voxel_colors = []

        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                for k in range(self.dims[2]):
                    if self.grid_state[i, j, k] in states:
                        voxel_colors.append([0, 0, 0])
                        voxel_centers.append(self.get_voxel_center(np.array([i, j, k])))

        pcd.points = o3d.utility.Vector3dVector(np.array(voxel_centers))
        pcd.colors = o3d.utility.Vector3dVector(np.array(voxel_colors))

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=self.min_bound)

        o3d.visualization.draw_geometries([pcd, mesh_frame])

    def visualize_grid(self):
        "visualizes voxelgrid as points that represent the voxel centers with the respected color for each state"
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
                    elif self.grid_state[i, j, k] == VoxelState.Occupied:
                        voxel_colors.append([0.1, 0.1, 0.1])
                        voxel_centers.append(self.get_voxel_center(np.array([i, j, k])))

        pcd.points = o3d.utility.Vector3dVector(np.array(voxel_centers))
        pcd.colors = o3d.utility.Vector3dVector(np.array(voxel_colors))

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=self.min_bound)

        o3d.visualization.draw_geometries([pcd, mesh_frame])

        return pcd


def is_ceiling(normal, z_thresh=0.9, direction=-1):
    return normal[2] < z_thresh * direction


def is_floor(normal, z_thresh=0.9, direction=1):
    return normal[2] > z_thresh * direction


def is_wall(normal, z_thresh=0.1):
    return normal[2] < z_thresh


def voxelize_mesh(_mesh):

    mesh = copy.deepcopy(_mesh)

    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    mesh.compute_triangle_normals(normalized=True)
    triangle_normals = np.asarray(mesh.triangle_normals)
    triangle_centers = np.mean(vertices[triangles], axis=1)

    triangle_centers_tree = KDTree(triangle_centers)

    voxel_size = 0.05
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)
    voxels = voxel_grid.get_voxels()

    mesh.compute_vertex_normals()
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=voxel_grid.get_min_bound())
    # o3d.visualization.draw_geometries([mesh_frame, room_mesh])

    seg = VoxelGrid(voxel_grid)

    for voxel in tqdm(voxels, desc="Assigning mesh normals to voxels"):
        voxel_center = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)

        # find near triangles
        _, idx = triangle_centers_tree.query(voxel_center, 5)
        for tri_id in idx:
            triangle = triangles[tri_id]
            if seg.triangle_intersects_voxel(voxel.grid_index, vertices[triangle]):
                normal = triangle_normals[tri_id]

        if is_ceiling(normal):
            seg.set_state(voxel.grid_index, VoxelState.Ceiling)
        elif is_floor(normal):
            seg.set_state(voxel.grid_index, VoxelState.Floor)
        elif is_wall(normal):
            seg.set_state(voxel.grid_index, VoxelState.Wall)
        else:
            seg.set_state(voxel.grid_index, VoxelState.Occupied)

    reconstruction(seg)
    seg.extract_inliers_mesh(mesh, VoxelState.Occupied)
    seg.extract_inliers_mesh(mesh, VoxelState.Wall)
    seg.extract_inliers_mesh(mesh, VoxelState.Ceiling)
    seg.extract_inliers_mesh(mesh, VoxelState.Floor)


def voxelize_pcd(pcd):

    room_pcd = copy.deepcopy(pcd)

    voxel_size = 0.05
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(room_pcd, voxel_size=voxel_size)

    room_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(20), fast_normal_computation=False)

    points = np.asarray(room_pcd.points)
    normals = np.asarray(room_pcd.normals)

    point_cloud_tree = KDTree(points)

    seg = VoxelGrid(voxel_grid)

    # Iterate through each voxel in the voxel grid to assign a normal
    for voxel in tqdm(voxel_grid.get_voxels(), desc="Assigning pcd normals to voxels"):
        voxel_index = voxel.grid_index

        # Find the points within this voxel
        voxel_center = np.array(voxel_grid.get_voxel_center_coordinate(voxel_index))
        _, idx = point_cloud_tree.query(voxel_center, 50)
        point_indices_mask = np.all(np.floor((points[idx] - np.asarray(voxel_grid.origin)) / voxel_size) == voxel_index, axis=1).astype(int)

        if len(point_indices_mask) > 0:
            normal = np.mean(normals[idx][point_indices_mask], axis=0)  # Calculate the mean normal for the points in this voxel
            normal /= np.linalg.norm(normal)  # Normalize the mean normal

            if normal[2] < -0.9 or normal[2] > 0.9:
                seg.set_state(voxel.grid_index, VoxelState.Ceiling)
            elif is_wall(normal):
                seg.set_state(voxel.grid_index, VoxelState.Wall)
            else:
                seg.set_state(voxel.grid_index, VoxelState.Occupied)

    reconstruction(seg)
    print("reconstruction done")
    seg.extract_inliers_pcd(room_pcd, VoxelState.Occupied)
    seg.extract_inliers_pcd(room_pcd, VoxelState.Wall)
    seg.extract_inliers_pcd(room_pcd, VoxelState.Ceiling)
    seg.extract_inliers_pcd(room_pcd, VoxelState.Floor)


def reconstruction(voxelgrid: VoxelGrid):

    voxelgrid.filter_normals()

    # voxelgrid.visualize_grid_state([VoxelState.Ceiling])
    voxelgrid.ceiling_floor_detection()
    # voxelgrid.visualize_grid_state([VoxelState.Ceiling])

    voxelgrid.floor_detection()

    # voxelgrid.visualize_grid_state([VoxelState.Floor])
    voxelgrid.floor_refinements()
    # voxelgrid.visualize_grid_state([VoxelState.Floor])

    voxelgrid.wall_detection()
    # voxelgrid.visualize_grid_state([VoxelState.Occupied])
    voxelgrid.fill_walls()
    # voxelgrid.visualize_grid_state([VoxelState.Occupied])
    # voxelgrid.visualize_grid_state([VoxelState.Occupied, VoxelState.Floor])

    voxelgrid.visualize_grid_state([VoxelState.Occupied])
    voxelgrid.visualize_grid_state([VoxelState.Wall])
    voxelgrid.visualize_grid_state([VoxelState.Ceiling])
    voxelgrid.visualize_grid_state([VoxelState.Occupied])


if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    # A1
    print("A1 - load room mesh")
    room_mesh = o3d.io.read_triangle_mesh("dataset/area_1/RoomMesh.ply")
    o3d.visualization.draw_geometries([room_mesh])

    # A2
    print("A2 - Mesh room objects ")
    voxelize_mesh(room_mesh)

    # A3
    print("A3 - Mesh Sampling")
    # room_pcd = uniform_sample.sample_mesh_uniformly(room_mesh, int(1e6))
    room_pcd = o3d.io.read_point_cloud("dataset/area_1/RoomPointCloud_1e6.ply")

    # A4
    print("A4 - PointCloud plane detection")
    voxelize_pcd(room_pcd)
