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
import dbscan
import contours_polygon as cp
import visibility_path as vp
from definitions import *


class VoxelLabel:
    Empty = 0
    Occupied = 1
    Occupied_air = 2
    Ceiling = 3
    Floor = 4
    Wall = 5

    # used for visualization
    DarkFloor = 100
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

        print(self.dims)

        self.grid_labels = np.ones(self.dims, dtype=int) * (VoxelLabel.Empty)
        self.grid_normals = np.ones(self.dims, dtype=int) * (VoxelNormal.Undefined)
        self.object_clusters = np.ones(self.dims, dtype=int) * (-2)
        self.floor_path_planning = np.zeros_like(self.grid_labels)

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
        occ_mask = self.grid_normals == VoxelNormal.Undefined
        occ_idx = np.nonzero(occ_mask)

        for i in tqdm(range(len(occ_idx)), desc="Normal Filtering"):
            self.grid_normals[occ_idx[0][i], occ_idx[1][i], occ_idx[2][i]] = self.filter_3d_26([occ_idx[0][i], occ_idx[1][i], occ_idx[2][i]], d=2)

    def ceiling_floor_detection(self):
        vertical_normal_mask = self.grid_normals == VoxelNormal.Vertical

        labeled_grid, num_labels = nd.label(vertical_normal_mask, nd.generate_binary_structure(3, 3))

        for label_id in tqdm(range(1, num_labels + 1), "Ceiling & Floor Detection"):
            segment_mask = labeled_grid == label_id
            idx = np.nonzero(segment_mask)

            vertical_normal_2d_mask = np.any(segment_mask, axis=2)

            # plt.imshow(vertical_normal_2d_mask)
            # plt.show()

            occupied_area = np.sum(vertical_normal_2d_mask)

            if occupied_area < int(np.round(VERTICAL_NORMAL_MIN_AREA / self.voxel_size**2)):
                continue

            mean_height = np.mean(idx[2])

            if mean_height < self.dims[2] / 2:
                if mean_height < FLOOR_MAX_HEIGHT:
                    self.grid_labels[segment_mask] = VoxelLabel.Floor
                    if self.floor_height == 0:
                        self.floor_height = mean_height
                    else:
                        self.floor_height = (self.floor_height + mean_height) / 2
            else:
                self.grid_labels[segment_mask] = VoxelLabel.Ceiling

    def ceiling_refinements(self):

        for k in tqdm(range(self.dims[2]), desc="Ceiling Refinement"):
            for j in range(self.dims[1]):
                for i in range(self.dims[0]):
                    # No occupied on ceiling level
                    if self.grid_labels[i, j, k] == VoxelLabel.Occupied:
                        if self.filter_2d_8(grid=self.grid_labels[:, :, k], idx=[i, j], state=VoxelLabel.Ceiling, d=5):
                            self.grid_labels[i, j, k] = VoxelLabel.Ceiling

        # pass walls through ceiling
        wall_2d_mask = np.any(self.grid_labels == VoxelLabel.Wall, axis=2)
        # plt.imshow(wall_2d_mask)
        # plt.show()

        for k in range(self.dims[2]):
            ceiling_2d_mask_k = self.grid_labels[:, :, k] == VoxelLabel.Ceiling
            ceiling_wall_mask = np.logical_and(wall_2d_mask, ceiling_2d_mask_k)
            if ceiling_wall_mask.any():
                # plt.imshow(ceiling_wall_mask)
                # plt.show()
                self.grid_labels[:, :, k][ceiling_wall_mask] = VoxelLabel.Wall

    def floor_refinements(self):

        ceiling_2d_mask = np.any(self.grid_labels == VoxelLabel.Ceiling, axis=2)
        wall_2d_mask = np.any(self.grid_labels[:, :, : int(self.dims[2] / 2)] == VoxelLabel.Wall, axis=2)

        # fill empty floor that has Ceiling over it
        floor_2d_mask = np.any(self.grid_labels == VoxelLabel.Floor, axis=2)
        fill_mask = np.logical_and(ceiling_2d_mask, 1 - floor_2d_mask)
        self.grid_labels[:, :, np.round(self.floor_height).astype(int)][fill_mask] = VoxelLabel.Floor

        for k in tqdm(range(np.round(self.floor_height).astype(int), -1, -1), desc="Floor Refinement"):
            # fill occupied voxels on floor level
            floor_occupied_2d_mask = self.grid_labels[:, :, k] == VoxelLabel.Occupied
            self.grid_labels[:, :, k][floor_occupied_2d_mask] = VoxelLabel.Floor

            # pass walls half height through floor
            floor_2d_mask_k = self.grid_labels[:, :, k] == VoxelLabel.Floor
            floor_wall_mask = np.logical_and(wall_2d_mask, floor_2d_mask_k)
            self.grid_labels[:, :, k][floor_wall_mask] = VoxelLabel.Wall

        self.floor_path_planning[self.grid_labels == VoxelLabel.Floor] = 1

        floor_2d_mask = np.any(self.grid_labels == VoxelLabel.Floor, axis=2)
        # plt.imshow(floor_2d_mask)
        # plt.show()

        np.save("floor_2d_mask", floor_2d_mask)

    def wall_detection(self):

        def wall_plane_detection_ransac(mask):
            min_num_of_vox = int(np.round(MIN_WALL_AREA / self.voxel_size**2))

            wall_mask_initial = copy.copy(mask)

            while np.sum(wall_mask_initial) > min_num_of_vox:

                (normal, d), inlier_mask = ransac.ransac_voxel_grid(wall_mask=wall_mask_initial, distance_threshold=RANSAC_PLANE_DISTANCE_THRESHOLD)

                # visualize wall inliers
                # mpla = self.grid_labels[inlier_mask]
                # self.grid_labels[inlier_mask] = VoxelLabel.plane
                # self.visualize_grid_state([VoxelLabel.plane])
                # self.grid_labels[inlier_mask] = mpla

                if np.sum(inlier_mask) < min_num_of_vox:
                    break

                wall_mask_initial[inlier_mask] = 0

            # returns a mask which contains the voxels not included in a wall plane
            return wall_mask_initial

        horizontal_normal_mask = self.grid_normals == VoxelNormal.Horizontal

        labeled_grid, num_labels = nd.label(horizontal_normal_mask, nd.generate_binary_structure(3, 3))

        min_wall_volume = int(np.round(MIN_WALL_SEGMENT_VOLUME / self.voxel_size**3))

        for label_id in tqdm(range(1, num_labels + 1), "Wall Detection"):
            segment_mask = labeled_grid == label_id

            occupied_volume = np.sum(segment_mask)

            if occupied_volume < min_wall_volume:
                self.grid_labels[segment_mask] = VoxelLabel.Occupied
                continue
            self.grid_labels[segment_mask] = VoxelLabel.Wall

            # horizontal_normal_2d_mask = np.any(segment_mask, axis=2)
            # plt.imshow(horizontal_normal_2d_mask)
            # plt.show()

            not_wall_mask = wall_plane_detection_ransac(segment_mask)

        # wall_2d_mask = np.any(self.grid_labels == VoxelLabel.Wall, axis=2)
        # plt.imshow(wall_2d_mask)
        # plt.show()

        self.grid_labels[not_wall_mask] = VoxelLabel.Occupied

        # wall_2d_mask = np.any(self.grid_labels == VoxelLabel.Wall, axis=2)
        # plt.imshow(wall_2d_mask)
        # plt.show()

        # np.save("wall_2d_mask", wall_2d_mask)

    def wall_refinements(self):
        half_height = (self.dims[2] / 2).astype(int)
        wall_2d_mask = np.any(self.grid_labels[:, :, :half_height] == VoxelLabel.Wall, axis=2)

        for k in range(self.dims[2] - 1, 1, -1):
            occupied_2dk_mask = self.grid_labels[:, :, k] == VoxelLabel.Occupied
            self.grid_labels[:, :, k][np.logical_and(wall_2d_mask, occupied_2dk_mask)] = VoxelLabel.Wall

        for i in tqdm(range(self.dims[0]), "Wall Refinement"):
            for j in range(self.dims[1]):
                for k in range(self.dims[2] - 1, 1, -1):
                    if self.grid_labels[i, j, k] == VoxelLabel.Wall:
                        if self.grid_labels[i, j, k - 1] == VoxelLabel.Occupied:
                            self.grid_labels[i, j, k - 1] = VoxelLabel.Wall

    def occupied_refinement(self):
        ceiling_2d_idx = np.where(self.grid_labels == VoxelLabel.Ceiling)

        ceiling_2d_mask = np.full((self.dims[0], self.dims[1]), np.inf)

        np.minimum.at(ceiling_2d_mask, (ceiling_2d_idx[0], ceiling_2d_idx[1]), ceiling_2d_idx[2])

        ceiling_height = np.mean(ceiling_2d_mask[ceiling_2d_mask != np.inf]).astype(int)

        # plt.imshow(wall_2d_mask.astype(int))
        # plt.show()

        for k in tqdm(range(self.dims[2] - 1, 1, -1), desc="Occupied Refinement"):

            for i in range(self.dims[0]):
                for j in range(self.dims[1]):
                    if self.grid_labels[i, j, k] == VoxelLabel.Occupied:
                        # if ceiling_2d_mask[i, j] < (k + 1 / self.voxel_size):
                        #     self.grid_labels[i, j, k] = VoxelLabel.Occupied_air
                        #     continue
                        if k > ceiling_height - 1 / self.voxel_size:
                            self.grid_labels[i, j, k] = VoxelLabel.Occupied_air

    def object_clustering(self):

        pcd = self.get_centroids_pcd(self.grid_labels, [VoxelLabel.Occupied])

        occupied_idx = np.argwhere(self.grid_labels == VoxelLabel.Occupied)
        labels = dbscan.dbscan_3d(occupied_idx, eps=OBJECT_CLUSTER_EPS, min_pts=OBJECT_CLUSTER_MIN_PTS)

        for i in range(len(labels)):
            self.object_clusters[occupied_idx[i, 0], occupied_idx[i, 1], occupied_idx[i, 2]] = labels[i]

        max_label = labels.max()
        print("Found Objects:", max_label)

        # visualize objects clusters
        colors = plt.get_cmap("tab20b")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

        self.object_pcd = pcd
        return pcd
        # o3d.visualization.draw_geometries([pcd])

    def door_recognition(self):
        wall_mask = self.grid_labels[:, :, : int(self.dims[2] / 2)] == VoxelLabel.Wall
        # plt.imshow(wall_mask)
        # plt.show()

        i = 0

        ceiling_floor_wall_mask = np.argwhere(self.grid_labels == VoxelLabel.Ceiling)

        while i < 4:

            (plane_normal, plane_point), inlier_mask = ransac.ransac_voxel_grid(wall_mask=wall_mask, distance_threshold=2)

            distance = ransac.distance_from_plane(ceiling_floor_wall_mask, plane_normal, plane_point)
            inliers = ceiling_floor_wall_mask[distance < 1]

            # projected_grid = project_voxel_grid(wall_mask[inlier_mask], normal, d, (wall_mask, 100))

            # visualize wall inliers
            mpla = self.grid_labels[inliers]
            self.grid_labels[inliers] = VoxelLabel.plane
            self.visualize_grid_state([VoxelLabel.plane])
            self.grid_labels[inliers] = mpla

            wall_mask[inlier_mask] = 0

    def project_objects(self):
        object_2d_mask = np.any(self.object_clusters[:, :, :POINT_HEIGHT] >= 0, axis=2)
        # plt.imshow(object_2d_mask)
        # plt.show()

        np.save("object_2d_mask", object_2d_mask)

        for k in range(np.max(self.object_clusters) + 1):
            object_2d_mask = np.any(self.object_clusters[:, :, :POINT_HEIGHT] == k, axis=2)

            self.floor_path_planning[object_2d_mask] = 0

        labeled_array, num_features = nd.label(self.floor_path_planning)

        # Count the size of each component
        component_sizes = np.bincount(labeled_array.ravel())

        # Ignore the background (component 0)
        component_sizes[0] = 0

        # Find the label of the largest component
        largest_component_label = component_sizes.argmax()

        # Create a mask for the largest component
        largest_component_mask = labeled_array == largest_component_label

        # Create a new grid with only the largest component
        self.floor_path_planning = largest_component_mask.astype(int)

        mpla = self.grid_labels
        self.grid_labels[self.floor_path_planning == 1] = VoxelLabel.DarkFloor
        self.visualize_grid([VoxelLabel.Floor, VoxelLabel.DarkFloor], filename="images/room1/floor_dark.png", show=False)
        self.grid_labels = mpla

    def pick_point(self):

        pcd = self.get_centroids_pcd(self.floor_path_planning, [1])

        print("")
        print("Path Planing")
        print("1) Please pick at least 2 correspondences using [shift + left click]")
        print("   Press [shift + right click] to undo point picking")
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()

        points = vis.get_picked_points()
        points = np.asarray(pcd.points)[points]

        points = self.get_voxel_id(points)[:, :2]

        return points

    def path_planning(self):

        points = self.pick_point()

        # points = np.array([[16, 9], [105, 45]])

        floor_path_planning_2d = np.any(self.floor_path_planning, axis=2)
        # plt.imshow(floor_path_planning_2d)
        # plt.show()

        # np.save("floor_2d_path", floor_path_planning_2d)

        polygon = cp.get_floor_polygon(floor_path_planning_2d.T)

        path = points[0]

        for i in range(len(points) - 1):
            try:
                comp_path = vp.visibility_path(tuple(points[i]), tuple(points[i + 1]), polygon)
                path = np.vstack((path, comp_path))
            except:
                print(" Unable to Compute Path between:", points[i], points[i + 1])
                return

        # vp.plot_visibility_path(polygon, path)

        path = np.hstack((path, np.ones((path.shape[0], 1)) * (np.round(self.floor_height) + 2))).astype(int)

        path = self.get_voxel_center(path)

        dist = vp.total_distance(path)

        print("Computed Path (", dist, ") :")
        print(path)

        lines = o3d.geometry.LineSet()
        lines.points = o3d.utility.Vector3dVector(path)
        lines.lines = o3d.utility.Vector2iVector(np.array([[i, i + 1] for i in range(len(path) - 1)]))

        mpla = self.grid_labels
        self.grid_labels[self.floor_path_planning == 1] = VoxelLabel.DarkFloor

        self.visualize_grid([VoxelLabel.Floor, VoxelLabel.DarkFloor], [lines], filename="images/room1/path_no_obs.png", show=False)
        self.visualize_grid([VoxelLabel.Floor, VoxelLabel.DarkFloor], [lines, self.object_pcd], filename="images/room1/path_obj.png", show=True)

        self.grid_labels = mpla

        return lines

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
        return inlier_pcd

    def get_centroids_pcd(self, grid, states):
        pcd = o3d.geometry.PointCloud()
        voxel_centers = []
        voxel_colors = []

        # color_map = plt.cm.viridis(np.linspace(0, 1, len(states)))[:, :3]

        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                for k in range(self.dims[2]):
                    lbl = grid[i, j, k]
                    if lbl in states:
                        # voxel_colors.append(color_map[states.index(lbl)])
                        voxel_centers.append(self.get_voxel_center(np.array([i, j, k])))

        pcd.points = o3d.utility.Vector3dVector(np.array(voxel_centers))
        # pcd.colors = o3d.utility.Vector3dVector(np.array(voxel_colors)
        pcd.paint_uniform_color([0, 0, 0])

        return pcd

    def visualize_grid_state(self, states):
        "visualizes voxel centers that belong to the given states list"
        pcd = self.get_centroids_pcd(self.grid_labels, states=states)

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=self.min_bound)

        o3d.visualization.draw_geometries([pcd, mesh_frame])

    def visualize_grid(self, states=[], extras=[], filename=None, show=True):
        "visualizes voxelgrid as points that represent the voxel centers with the respected color for each state"
        pcd = o3d.geometry.PointCloud()
        voxel_centers = []
        voxel_colors = []
        st = True

        if states[0] != None:
            if not states:
                st = False

            for i in range(self.dims[0]):
                for j in range(self.dims[1]):
                    for k in range(self.dims[2]):
                        if self.grid_labels[i, j, k] == VoxelLabel.Ceiling and ((VoxelLabel.Ceiling in states and st == True) or st == False):
                            voxel_colors.append([1, 0, 0])
                            voxel_centers.append(self.get_voxel_center(np.array([i, j, k])))
                        if self.grid_labels[i, j, k] == VoxelLabel.Floor and ((VoxelLabel.Floor in states and st == True) or st == False):
                            if VoxelLabel.DarkFloor in states:
                                voxel_colors.append([0, 0.5, 0])
                            else:
                                voxel_colors.append([0, 1, 0])
                            voxel_centers.append(self.get_voxel_center(np.array([i, j, k])))
                        if self.grid_labels[i, j, k] == VoxelLabel.DarkFloor and (VoxelLabel.DarkFloor in states and st == True):
                            voxel_colors.append([0, 1, 0])
                            voxel_centers.append(self.get_voxel_center(np.array([i, j, k])))
                        if self.grid_labels[i, j, k] == VoxelLabel.Wall and ((VoxelLabel.Wall in states and st == True) or st == False):
                            voxel_colors.append([0.6, 0.6, 0.6])
                            voxel_centers.append(self.get_voxel_center(np.array([i, j, k])))
                        if self.grid_labels[i, j, k] == VoxelLabel.Occupied and ((VoxelLabel.Occupied in states and st == True) or st == False):
                            voxel_colors.append([0.1, 0.1, 0.1])
                            voxel_centers.append(self.get_voxel_center(np.array([i, j, k])))

            pcd.points = o3d.utility.Vector3dVector(np.array(voxel_centers))
            pcd.colors = o3d.utility.Vector3dVector(np.array(voxel_colors))

        geometries = [pcd] + extras

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=show)

        for g in geometries:
            vis.add_geometry(g)

        vis.get_render_option().show_coordinate_frame = True
        vis.get_render_option().point_size = 8
        vis.update_renderer()

        if show:
            vis.run()

        # param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        # o3d.io.write_pinhole_camera_parameters("viewpoint.json", param)

        if filename != None:
            time.sleep(1)
            vis.capture_screen_image(filename, do_render=True)

        vis.destroy_window()

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

    voxel_size = 0.05
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)
    voxels = voxel_grid.get_voxels()

    mesh.compute_vertex_normals()

    seg = VoxelGrid(voxel_grid)

    for voxel in tqdm(voxels, desc="Assigning mesh normals to voxels"):
        voxel_center = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)

        # find near triangles
        _, idx = triangle_centers_tree.query(voxel_center, 5)
        for tri_id in idx:
            triangle = triangles[tri_id]
            if seg.triangle_intersects_voxel(voxel.grid_index, vertices[triangle]):
                normal = triangle_normals[tri_id]

        if abs(normal[2]) > 0.9:
            seg.set_normal(voxel.grid_index, VoxelNormal.Vertical)
        elif abs(normal[2]) < 0.1:
            seg.set_normal(voxel.grid_index, VoxelNormal.Horizontal)

    reconstruction(seg)
    # seg.extract_inliers_mesh(mesh, VoxelState.Occupied)
    # seg.extract_inliers_mesh(mesh, VoxelState.Wall)
    # seg.extract_inliers_mesh(mesh, VoxelLabel.Ceiling)
    # seg.extract_inliers_mesh(mesh, VoxelState.Floor)


def voxelize_pcd(pcd, voxel_size=VOXEL_SIZE):
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
        _, idx = point_cloud_tree.query(voxel_center, 20)
        point_indices_mask = np.all(np.floor((points[idx] - np.asarray(voxel_grid.origin)) / voxel_size) == voxel_index, axis=1).astype(int)

        if len(point_indices_mask) > 0:
            normal = np.mean(normals[idx][point_indices_mask], axis=0)  # Calculate the mean normal for the points in this voxel
            normal /= np.linalg.norm(normal)  # Normalize the mean normal

            if abs(normal[2]) > 0.9:
                seg.set_normal(voxel.grid_index, VoxelNormal.Vertical)
            elif abs(normal[2]) < 0.1:
                seg.set_normal(voxel.grid_index, VoxelNormal.Horizontal)
            else:
                seg.set_state(voxel.grid_index, VoxelLabel.Occupied)

    reconstruction(seg)

    # seg.extract_inliers_pcd(room_pcd, VoxelLabel.Occupied)
    # seg.extract_inliers_pcd(room_pcd, VoxelLabel.Wall)
    # seg.extract_inliers_pcd(room_pcd, VoxelState.Ceiling)
    # seg.extract_inliers_pcd(room_pcd, VoxelLabel.Floor)


def reconstruction(voxelgrid: VoxelGrid):
    voxelgrid.filter_normals()

    # detection
    voxelgrid.ceiling_floor_detection()
    voxelgrid.wall_detection()

    voxelgrid.visualize_grid([VoxelLabel.Floor, VoxelLabel.Wall], filename="images/room1/Floor_Wall_Det.png", show=False)

    # refinements
    voxelgrid.ceiling_refinements()
    voxelgrid.floor_refinements()
    voxelgrid.wall_refinements()
    voxelgrid.occupied_refinement()

    voxelgrid.visualize_grid([VoxelLabel.Floor, VoxelLabel.Wall], filename="images/room1/Floor_Wall_Ref.png", show=False)
    voxelgrid.visualize_grid([VoxelLabel.Floor, VoxelLabel.Wall, VoxelLabel.Occupied], filename="images/room1/Floor_Wall_Occ_Ref.png", show=False)

    print("####_RECONSTRUCTION_DONE_####")

    objects_pcd = voxelgrid.object_clustering()

    voxelgrid.visualize_grid([VoxelLabel.Floor, VoxelLabel.Wall], [objects_pcd], filename="images/room1/Floor_Wall_Obj.png", show=False)

    voxelgrid.project_objects()

    while True:
        voxelgrid.path_planning()

    # voxelgrid.visualize_grid_state([VoxelLabel.Ceiling])
    # voxelgrid.visualize_grid_state([VoxelLabel.Occupied])
    # voxelgrid.visualize_grid_state([VoxelLabel.Wall, VoxelLabel.Floor, VoxelLabel.Occupied])
    # voxelgrid.visualize_grid_state([VoxelLabel.Floor])

    # voxelgrid.visualize_grid()


if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    # # A1
    # print("A1 - load room mesh")
    room_mesh = o3d.io.read_triangle_mesh("dataset/area_1/RoomMesh.ply")
    # room_mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([room_mesh])

    # # A2
    # print("A2 - Mesh room objects ")
    # start = time.time()
    # voxelize_mesh(room_mesh)
    # print("MESH-voxels time: ", time.time() - start)

    # # A3
    # print("A3 - Mesh Sampling")
    room_pcd = uniform_sample.sample_mesh_uniformly(room_mesh, int(1e5))

    room_pcd = o3d.io.read_point_cloud("dataset/area_1/RoomPointCloud_1e6.ply")
    # #  crop for vis
    # points = np.asarray(room_pcd.points)
    # room_pcd.points = o3d.utility.Vector3dVector(points[points[:, 2] < 2.5])
    # room_pcd = o3d.io.read_point_cloud("dataset/Stanford3dDataset_v1.2/Area_3/office_1/office_1.txt", format="xyz")
    # room_pcd = o3d.io.read_point_cloud("dataset/Stanford3dDataset_v1.2/Area_1/conferenceRoom_1/conferenceRoom_1.txt", format="xyz")
    # room_pcd = o3d.io.read_point_cloud("dataset/area_3/Area_3/conferenceRoom_1/conferenceRoom_1.txt", format="xyz")
    o3d.visualization.draw_geometries([room_pcd])

    # room_pcd = room_pcd.rotate(room_pcd.get_rotation_matrix_from_xyz(np.array([[-np.pi / 2, 0, 0]]).T))

    # o3d.visualization.draw([room_pcd])

    # # A4
    # print("A4 - PointCloud plane detection")

    start = time.time()
    voxelize_pcd(room_pcd, 0.05)
    print("PCD-voxels time: ", time.time() - start)
