import open3d as o3d
import copy
import numpy as np
from tqdm import tqdm
from scipy.spatial import KDTree
import scipy.ndimage as nd
from matplotlib import pyplot as plt
import uniform_sample
import time
import ransac


class VoxelLabel:
    Empty = 0
    Occupied = 1
    Occupied_air = 2
    Ceiling = 3
    Floor = 4
    Wall = 5
    plane = 10


class VoxelNormal:
    Horizontal = 0
    Vertical = 1
    Undefined = -1


class VoxelGrid:

    def __init__(self, voxelgrid) -> None:
        self.voxel_grid = voxelgrid

        self.min_bound = voxelgrid.get_min_bound()
        self.max_bound = voxelgrid.get_max_bound()
        self.voxel_size = voxelgrid.voxel_size
        self.dims = np.ceil((self.max_bound - self.min_bound) / self.voxel_size).astype(int)

        self.grid_labels = np.ones(self.dims, dtype=int) * (VoxelLabel.Empty)
        self.grid_normals = np.ones(self.dims, dtype=int) * (VoxelNormal.Undefined)

        self.floor_height = 0

        for v in voxelgrid.get_voxels():
            self.grid_labels[v.grid_index[0], v.grid_index[1], v.grid_index[2]] = VoxelLabel.Occupied

    def get_voxel_center(self, idx):
        return self.min_bound + idx * self.voxel_size + [self.voxel_size / 2, self.voxel_size / 2, self.voxel_size / 2]

    def get_voxel_id(self, voxel_center):
        id = np.round((voxel_center - self.min_bound - [self.voxel_size / 2, self.voxel_size / 2, self.voxel_size / 2]) / self.voxel_size).astype(int)
        return id

    def set_state(self, idx, state):
        self.grid_labels[idx[0], idx[1], idx[2]] = state

    def set_normal(self, idx, normal):
        self.grid_normals[idx[0], idx[1], idx[2]] = normal

    def filter_normals(self):
        # print("Normal Filtering")
        # ceiling_mask = self.grid_state == VoxelState.Ceiling
        # ceiling_idx = np.nonzero(ceiling_mask)

        # change_mask = np.zeros_like(self.grid_state)

        # for i, v in enumerate(self.grid_state[ceiling_mask]):
        #     self.grid_state[ceiling_idx[0][i], ceiling_idx[1][i], ceiling_idx[2][i]] = self.filter_3d_26(
        #         [ceiling_idx[0][i], ceiling_idx[1][i], ceiling_idx[2][i]]
        #     )

        # floor_mask = self.grid_state == VoxelState.Floor
        # floor_idx = np.nonzero(floor_mask)

        # for i, v in enumerate(self.grid_state[floor_mask]):
        #     self.grid_state[floor_idx[0][i], floor_idx[1][i], floor_idx[2][i]] = self.filter_3d_26(
        #         [floor_idx[0][i], floor_idx[1][i], floor_idx[2][i]]
        #     )

        # wall_mask = self.grid_state == VoxelState.Wall
        # wall_idx = np.nonzero(wall_mask)

        # for i, v in enumerate(self.grid_state[wall_mask]):
        #     self.grid_state[wall_idx[0][i], wall_idx[1][i], wall_idx[2][i]] = self.filter_3d_26([wall_idx[0][i], wall_idx[1][i], wall_idx[2][i]])

        # occ_mask = self.grid_labels == VoxelLabel.Occupied
        occ_mask = self.grid_normals == VoxelNormal.Undefined
        occ_idx = np.nonzero(occ_mask)

        for i in tqdm(range(len(occ_idx)), desc="Normal Filtering"):
            self.grid_normals[occ_idx[0][i], occ_idx[1][i], occ_idx[2][i]] = self.filter_3d_26([occ_idx[0][i], occ_idx[1][i], occ_idx[2][i]], d=2)

    def ceiling_floor_detection(self, min_area=1):
        vertical_normal_mask = self.grid_normals == VoxelNormal.Vertical

        labeled_grid, num_labels = nd.label(vertical_normal_mask, nd.generate_binary_structure(3, 3))

        for label_id in tqdm(range(1, num_labels + 1), "Ceiling & Floor Detection"):
            segment_mask = labeled_grid == label_id
            idx = np.nonzero(segment_mask)

            vertical_normal_2d_mask = np.any(segment_mask, axis=2)

            # plt.imshow(vertical_normal_2d_mask)
            # plt.show()

            occupied_area = np.sum(vertical_normal_2d_mask)

            if occupied_area < int(np.round(min_area / self.voxel_size**2)):
                continue

            mean_height = np.mean(idx[2])

            if mean_height < self.dims[2] / 2:
                if mean_height < 5:
                    self.grid_labels[segment_mask] = VoxelLabel.Floor
                    self.floor_height = (self.floor_height + mean_height) / 2
            else:
                self.grid_labels[segment_mask] = VoxelLabel.Ceiling

    def ceiling_refinements(self):

        for k in tqdm(range(self.dims[2]), desc="Ceiling Refinements"):
            for j in range(self.dims[1]):
                for i in range(self.dims[0]):
                    # Pass walls through Ceiling
                    if k > 0:
                        if (self.grid_labels[i, j, k] == VoxelLabel.Ceiling) and self.grid_labels[i, j, k - 1] == VoxelLabel.Wall:
                            self.grid_labels[i, j, k] = VoxelLabel.Wall

                    # No occupied on ceiling level
                    if self.grid_labels[i, j, k] == VoxelLabel.Occupied:
                        if self.filter_2d_8(grid=self.grid_labels[:, :, k], idx=[i, j], state=VoxelLabel.Ceiling, d=5):
                            self.grid_labels[i, j, k] = VoxelLabel.Ceiling

    def floor_refinements(self):
        zero = -100

        # fill empty floor that has Ceiling over it
        ceiling_2d_mask = np.any(self.grid_labels == VoxelLabel.Ceiling, axis=2)
        floor_2d_mask = np.any(self.grid_labels == VoxelLabel.Floor, axis=2)

        fill_mask = np.logical_or(ceiling_2d_mask, floor_2d_mask)

        self.grid_labels[:, :, np.round(self.floor_height).astype(int)][fill_mask] = VoxelLabel.Floor

        # fill occupied voxels on floor level
        floor_occupied_2d_mask = self.grid_labels[:, :, np.round(self.floor_height).astype(int)] == VoxelLabel.Occupied

        for k in range(np.round(self.floor_height).astype(int), -1, -1):
            self.grid_labels[:, :, k][floor_occupied_2d_mask] = VoxelLabel.Floor

        # plt.imshow(fill_mask)
        # plt.show()

    def wall_detection(self, min_area=2):

        def wall_plane_detection_ransac(mask, min_area_=min_area):
            min_num_of_vox = int(np.round(min_area_ / self.voxel_size**2))

            wall_mask = copy.copy(mask)

            while np.sum(wall_mask) > min_num_of_vox:

                (normal, d), inlier_mask = ransac.ransac_voxel_grid(wall_mask=wall_mask, distance_threshold=1.5)

                # self.grid_labels[inlier_mask] = VoxelLabel.plane
                # self.visualize_grid_state([VoxelLabel.plane])

                if np.sum(inlier_mask) < min_num_of_vox:
                    break

                wall_mask[inlier_mask] = 0

            self.grid_labels[wall_mask] = VoxelLabel.Occupied

        horizontal_normal_mask = self.grid_normals == VoxelNormal.Horizontal

        labeled_grid, num_labels = nd.label(horizontal_normal_mask, nd.generate_binary_structure(3, 3))

        for label_id in tqdm(range(1, num_labels + 1), "Wall Detection"):
            segment_mask = labeled_grid == label_id

            occupied_volume = np.sum(segment_mask)

            if occupied_volume < int(np.round(0.5 / self.voxel_size**3)):
                self.grid_labels[segment_mask] = VoxelLabel.Occupied
                continue
            self.grid_labels[segment_mask] = VoxelLabel.Wall

            # horizontal_normal_2d_mask = np.any(segment_mask, axis=2)

            # plt.imshow(horizontal_normal_2d_mask)
            # plt.show()

            wall_plane_detection_ransac(segment_mask)

        # vertical_normal_2d_mask = np.any(self.grid_labels == VoxelLabel.Wall, axis=2)
        # plt.imshow(vertical_normal_2d_mask)
        # plt.show()

        # wall_plane_detection_ransac()

    def wall_refinements(self):
        for i in tqdm(range(self.dims[0]), "Wall Refinement"):
            for j in range(self.dims[1]):
                for k in range(self.dims[2] - 1, 1, -1):
                    if self.grid_labels[i, j, k] == VoxelLabel.Wall:

                        if self.grid_labels[i, j, k - 1] == VoxelLabel.Occupied:
                            self.grid_labels[i, j, k - 1] = VoxelLabel.Wall

                        if self.grid_labels[i, j, k - 1] == VoxelLabel.Floor:
                            self.grid_labels[i, j, k - 1] = VoxelLabel.Wall

    def occupied_detection(self):
        ceiling_2d_idx = np.where(self.grid_labels == VoxelLabel.Ceiling)

        ceiling_2d_mask = np.full((self.dims[0], self.dims[1]), np.inf)

        np.minimum.at(ceiling_2d_mask, (ceiling_2d_idx[0], ceiling_2d_idx[1]), ceiling_2d_idx[2])

        ceiling_height = np.mean(ceiling_2d_mask[ceiling_2d_mask != np.inf]).astype(int)

        # plt.imshow(ceiling_2d_mask.astype(int))
        # plt.show()

        for k in tqdm(range(self.dims[2] - 1, 1, -1), desc="Occupied Detection"):
            for i in range(self.dims[0]):
                for j in range(self.dims[1]):
                    if self.grid_labels[i, j, k] == VoxelLabel.Occupied:
                        if ceiling_2d_mask[i, j] < (k + 1 / self.voxel_size):
                            self.grid_labels[i, j, k] = VoxelLabel.Occupied_air
                            continue
                        if k > ceiling_height - 1 / self.voxel_size:
                            self.grid_labels[i, j, k] = VoxelLabel.Occupied_air

                        # if self.filter_3d_26([i, j, k], VoxelState.Wall, threshold=1, d=5):
                        #     self.grid_state[i, j, k] = VoxelState.Wall
                    # if self.grid_state[i, j, k] == VoxelState.Wall and (k < ceiling_2d_mask[i, j] < np.inf):
                    #     self.grid_state[i, j, k] = VoxelState.Occupied

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

    def filter_3d_26(self, idx, state=None, threshold=3, d=1):
        states_instances_dict = np.zeros((10,))

        num_neighbors = 0
        for di in range(-d, d + 1):
            for dj in range(-d, d + 1):
                for dk in range(-d, d + 1):
                    if (
                        (di != 0 or dj != 0 or dk != 0)
                        and 0 <= idx[0] + di < self.grid_labels.shape[0]
                        and 0 <= idx[1] + dj < self.grid_labels.shape[1]
                        and 0 <= idx[2] + dk < self.grid_labels.shape[2]
                    ):
                        states_instances_dict[self.grid_labels[idx[0] + di, idx[1] + dj, idx[2] + dk].astype(int)] += 1

                        if self.grid_labels[idx[0] + di, idx[1] + dj, idx[2] + dk] == state:
                            num_neighbors += 1

        if state == None:
            states_instances_dict[0] = 0  # Dont count empty
            return np.argmax(states_instances_dict)

        if num_neighbors > threshold:
            return True
        return False

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
                return self.grid_labels[voxel[0], voxel[1], voxel[2]] == state
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
                return self.grid_labels[voxel[0], voxel[1], voxel[2]] == state
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
                    if self.grid_labels[i, j, k] in states:
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
                    if self.grid_labels[i, j, k] == VoxelLabel.Ceiling:
                        voxel_colors.append([1, 0, 0])
                        voxel_centers.append(self.get_voxel_center(np.array([i, j, k])))
                    elif self.grid_labels[i, j, k] == VoxelLabel.Floor:
                        voxel_colors.append([0, 1, 0])
                        voxel_centers.append(self.get_voxel_center(np.array([i, j, k])))
                    elif self.grid_labels[i, j, k] == VoxelLabel.Wall:
                        voxel_colors.append([0.6, 0.6, 0.6])
                        voxel_centers.append(self.get_voxel_center(np.array([i, j, k])))
                    elif self.grid_labels[i, j, k] == VoxelLabel.Occupied:
                        voxel_colors.append([0.1, 0.1, 0.1])
                        voxel_centers.append(self.get_voxel_center(np.array([i, j, k])))

        pcd.points = o3d.utility.Vector3dVector(np.array(voxel_centers))
        pcd.colors = o3d.utility.Vector3dVector(np.array(voxel_colors))

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=self.min_bound)

        o3d.visualization.draw_geometries([pcd, mesh_frame])

        return pcd


def voxelize_mesh(_mesh):
    print("####_VOXELIZE_MESH_####")

    mesh = copy.deepcopy(_mesh)

    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    mesh.compute_triangle_normals(normalized=True)
    triangle_normals = np.asarray(mesh.triangle_normals)
    triangle_centers = np.mean(vertices[triangles], axis=1)

    triangle_centers_tree = KDTree(triangle_centers)
    # triangle_centers_tree = o3d.geometry.KDTreeFLANN(triangle_centers)

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

        # if is_ceiling(normal):
        #     seg.set_state(voxel.grid_index, VoxelLabel.Ceiling)
        # elif is_floor(normal):
        #     seg.set_state(voxel.grid_index, VoxelLabel.Floor)
        # elif is_wall(normal):
        #     seg.set_state(voxel.grid_index, VoxelLabel.Wall)
        # else:
        #     seg.set_state(voxel.grid_index, VoxelLabel.Occupied)

        if abs(normal[2]) > 0.9:
            # seg.set_state(voxel.grid_index, VoxelLabel.Ceiling)
            seg.set_normal(voxel.grid_index, VoxelNormal.Vertical)
        elif abs(normal[2]) < 0.1:
            # seg.set_state(voxel.grid_index, VoxelLabel.Wall)
            seg.set_normal(voxel.grid_index, VoxelNormal.Horizontal)
        # else:
        #     seg.set_state(voxel.grid_index, VoxelLabel.Occupied)

    reconstruction(seg)
    # seg.extract_inliers_mesh(mesh, VoxelState.Occupied)
    # seg.extract_inliers_mesh(mesh, VoxelState.Wall)
    seg.extract_inliers_mesh(mesh, VoxelLabel.Ceiling)
    # seg.extract_inliers_mesh(mesh, VoxelState.Floor)


def voxelize_pcd(pcd, voxel_size=0.05):
    print("####_VOXELIZE_POINTCLOUD_####")

    room_pcd = copy.deepcopy(pcd)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(room_pcd, voxel_size=voxel_size)

    room_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(20), fast_normal_computation=True)

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

            if abs(normal[2]) > 0.9:
                # seg.set_state(voxel.grid_index, VoxelLabel.Ceiling)
                seg.set_normal(voxel.grid_index, VoxelNormal.Vertical)
            elif abs(normal[2]) < 0.1:
                # seg.set_state(voxel.grid_index, VoxelLabel.Wall)
                seg.set_normal(voxel.grid_index, VoxelNormal.Horizontal)
            else:
                seg.set_state(voxel.grid_index, VoxelLabel.Occupied)

    reconstruction(seg)

    seg.extract_inliers_pcd(room_pcd, VoxelLabel.Occupied)
    seg.extract_inliers_pcd(room_pcd, VoxelLabel.Wall)
    # seg.extract_inliers_pcd(room_pcd, VoxelState.Ceiling)
    # seg.extract_inliers_pcd(room_pcd, VoxelLabel.Floor)


def reconstruction(voxelgrid: VoxelGrid):
    voxelgrid.filter_normals()

    # detection
    voxelgrid.ceiling_floor_detection()
    voxelgrid.wall_detection()

    # refinements
    voxelgrid.ceiling_refinements()
    voxelgrid.floor_refinements()
    voxelgrid.wall_refinements()

    voxelgrid.occupied_detection()

    print("####_RECONSTRUCTION_DONE_####")
    voxelgrid.visualize_grid_state([VoxelLabel.Ceiling])
    voxelgrid.visualize_grid_state([VoxelLabel.Occupied])
    voxelgrid.visualize_grid_state([VoxelLabel.Wall])
    voxelgrid.visualize_grid_state([VoxelLabel.Floor])

    voxelgrid.visualize_grid()


if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    # # A1
    # print("A1 - load room mesh")
    room_mesh = o3d.io.read_triangle_mesh("dataset/area_1/RoomMesh.ply")
    # # o3d.visualization.draw_geometries([room_mesh])

    # # A2
    # print("A2 - Mesh room objects ")
    # start = time.time()
    # voxelize_mesh(room_mesh)
    # print("MESH-voxels time: ", time.time() - start)

    # # A3
    # print("A3 - Mesh Sampling")
    # room_pcd = uniform_sample.sample_mesh_uniformly(room_mesh, int(1e6))
    room_pcd = o3d.io.read_point_cloud("dataset/area_1/RoomPointCloud_1e6.ply")
    # room_pcd = o3d.io.read_point_cloud("dataset/area_3/Area_3/office_6/office_6.txt", format="xyz")
    # o3d.visualization.draw_geometries([room_pcd])

    # # A4
    # print("A4 - PointCloud plane detection")

    start = time.time()
    voxelize_pcd(room_pcd, 0.07)
    print("PCD-voxels time: ", time.time() - start)
