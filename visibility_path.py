import numpy as np
from shapely.geometry import *
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import contours_polygon as cp

import time

DISTANCE_TOLERANCE = 1e-6


def get_next_vert(i, polygon, direction=1):
    i = i + direction

    if i > len(polygon.coords) - 2:
        i = 0
    elif i < 0:
        i = len(polygon.coords) - 2

    return i, polygon.coords[i]


def get_closest_intersection(p1, p2, polygons):
    p1p2 = LineString([p1, p2])

    intersection = p1p2.intersection(polygons)

    # Start point of the line
    start_point = Point(p1)
    end_point = Point(p2)

    # Extract intersection points
    intersection_points = []

    if isinstance(intersection, MultiPoint):
        intersection_points.extend(intersection.geoms)
    elif isinstance(intersection, LineString):
        intersection_points.extend([Point(coord) for coord in intersection.coords])
    elif isinstance(intersection, MultiLineString):
        for geom in intersection.geoms:
            if isinstance(geom, LineString):
                intersection_points.extend([Point(coord) for coord in geom.coords])

    elif isinstance(intersection, GeometryCollection):
        # If the intersection is a collection, extract points from the collection
        for geom in intersection.geoms:
            if isinstance(geom, Point):
                intersection_points.append(geom)
            elif isinstance(geom, LineString):
                intersection_points.extend([Point(coord) for coord in geom.coords])

    # Find the closest intersection point to the start of the line
    closest_point = None
    min_distance = float("inf")

    for point in intersection_points:
        start_distance = start_point.distance(point)
        end_distance = end_point.distance(point)
        if start_distance < min_distance and start_distance > DISTANCE_TOLERANCE and end_distance > DISTANCE_TOLERANCE:
            min_distance = start_distance
            closest_point = point

    return closest_point


def total_distance(points):
    # Convert points to a numpy array for easier manipulation
    points = np.array(points)
    # Calculate the differences between consecutive points
    differences = np.diff(points, axis=0)
    # Compute the Euclidean distance for each pair of points
    distances = np.sqrt(np.sum(differences**2, axis=1))
    # Sum the distances to get the total distance
    total_dist = np.sum(distances)
    return total_dist


def visibility_path(p1, p2, polygon_with_holes, d=0):
    # print("")
    # print("visipath", p1, p2, d)

    path = np.array([p1])
    # p1p2_line = LineString([p1, p2])

    closest_intersection = get_closest_intersection(p1, p2, polygon_with_holes)

    # print("closest intersection :", closest_intersection)

    if d == 5:
        closest_intersection = None

    linearrings = [polygon_with_holes.exterior] + list(polygon_with_holes.interiors)

    # print(linearrings)

    if closest_intersection != None:
        # find intersecting polygon
        for poly in linearrings:
            if poly.distance(closest_intersection) < DISTANCE_TOLERANCE:
                ring = poly
                break

        # print(ring)

        # find intersecting edge of polygon
        for i in range(len(ring.coords) - 1):
            edge = LineString([ring.coords[i], ring.coords[i + 1]])

            if p1 == ring.coords[i]:
                icw, vcw = get_next_vert(i, ring, 1)
                iccw, vccw = get_next_vert(i, ring, -1)
                break
            elif edge.distance(closest_intersection) < DISTANCE_TOLERANCE:
                icw, vcw = get_next_vert(i, ring, 1)
                iccw = i
                vccw = ring.coords[iccw]
                break

        p1_vcw_path = visibility_path(p1, vcw, polygon_with_holes, d + 1)
        p1_vccw_path = visibility_path(p1, vccw, polygon_with_holes, d + 1)

        vcw_p2_line = LineString([vcw, p2])
        vccw_p2_line = LineString([vccw, p2])

        while vcw_p2_line.crosses(ring):
            icw, vcw = get_next_vert(icw, ring, 1)
            p1_vcw_path = np.vstack((p1_vcw_path, vcw))

            for i in range(len(p1_vcw_path) - 2, -1, -1):
                p_vcw_line = LineString([p1_vcw_path[i], vcw])

                if not p_vcw_line.crosses(ring) and polygon_with_holes.contains(p_vcw_line):  # ayto to gamidi
                    # print("mpla")
                    p_vcw_path = visibility_path(tuple(p1_vcw_path[i]), vcw, polygon_with_holes, d + 1)
                    p_vcw_path = np.vstack((p1_vcw_path[:i], p_vcw_path))

                    if total_distance(p_vcw_path) < total_distance(p1_vcw_path):
                        p1_vcw_path = p_vcw_path

            vcw_p2_line = LineString([vcw, p2])

        while vccw_p2_line.crosses(ring):  # or polygon.contains(vccw_p2_line):
            iccw, vccw = get_next_vert(iccw, ring, -1)
            p1_vccw_path = np.vstack((p1_vccw_path, vccw))

            for i in range(len(p1_vccw_path) - 2, -1, -1):
                p_vccw_line = LineString([p1_vccw_path[i], vccw])

                if not p_vccw_line.crosses(ring) and polygon_with_holes.contains(p_vccw_line):
                    # print("mpla")
                    p_vccw_path = visibility_path(tuple(p1_vccw_path[i]), vccw, polygon_with_holes, d + 1)
                    p_vccw_path = np.vstack((p1_vccw_path[:i], p_vccw_path))

                    if total_distance(p_vccw_path) < total_distance(p1_vccw_path):
                        p1_vccw_path = p_vccw_path

            vccw_p2_line = LineString([vccw, p2])

        vcw_p2_path = visibility_path(vcw, p2, polygon_with_holes, d + 1)
        vccw_p2_path = visibility_path(vccw, p2, polygon_with_holes, d + 1)

        vcw_p2_path = np.delete(vcw_p2_path, 0, axis=0)
        vccw_p2_path = np.delete(vccw_p2_path, 0, axis=0)

        p1p2cw_path = np.vstack((p1_vcw_path, vcw_p2_path))
        p1p2ccw_path = np.vstack((p1_vccw_path, vccw_p2_path))

        if total_distance(p1p2cw_path) <= total_distance(p1p2ccw_path):
            path = p1p2cw_path
        else:
            path = p1p2ccw_path

        # path = p1p2cw_path

    else:
        path = np.vstack((path, p2))

    # print(path)

    return path.astype(int)


def plot_visibility_path(polygon, path):

    fig, ax = plt.subplots()

    # Plot polygons

    x, y = polygon.exterior.xy
    plt.plot(x, y, color="red")

    for ring in polygon.interiors:
        x, y = ring.xy
        plt.plot(x, y, color="blue")

    # Plot visibility edges
    for i in range(len(path) - 1):
        x_values = [path[i][0], path[i + 1][0]]
        y_values = [path[i][1], path[i + 1][1]]
        ax.plot(x_values, y_values, color="green", linestyle="--")

    ax.set_aspect("equal", "box")
    plt.show()


def is_collinear(p1, p2, p3):
    """Check if three points are collinear."""
    return (p3.y - p1.y) * (p2.x - p1.x) == (p2.y - p1.y) * (p3.x - p1.x)


def remove_collinear_points(polygon):
    """Remove collinear points from a polygon."""
    if not isinstance(polygon, Polygon):
        raise TypeError("Input must be a Shapely Polygon")

    ring = LinearRing(polygon.exterior.coords)
    non_collinear_coords = []

    for i in range(len(ring.coords) - 2):
        p1 = Point(ring.coords[i])
        p2 = Point(ring.coords[i + 1])
        p3 = Point(ring.coords[i + 2])

        if not is_collinear(p1, p2, p3):
            non_collinear_coords.append(ring.coords[i + 1])

    # Adding the first point and the last point
    non_collinear_coords.insert(0, ring.coords[0])
    non_collinear_coords.append(ring.coords[-1])

    return Polygon(non_collinear_coords)


def polygons_preprocessing(polygon_points):

    polygons = []

    for poly in polygon_points:
        polygons.append(Polygon(poly))

    merged_polygon = unary_union(polygons)

    polygons = []

    for poly in merged_polygon.geoms:
        poly = remove_collinear_points(poly)
        polygons.append(poly)

    return polygons


if __name__ == "__main__":
    polygons = []
    path = []

    # # SECTION - Simple Test
    polygon1 = [(1, 1), (2, 2), (3, 4.5), (1, 4), (1, 1)]
    polygon2 = [(6.5, 2), (8, 2), (7, 5)]
    polygon_box = Polygon([(0, 0), (4, 0), (4, 6), (6, 5), (6, 0), (10, 0), (10, 10), (0, 10), (0, 0)], [polygon1, polygon2])

    # polygons = [polygon_box]

    start = (180, 30)
    end = (200, 120)

    # line = LineString([(1, 1), (3, 4.5)])
    path = [start, end]

    # ring = Polygon(polygon1)

    # print(ring.contains(line))
    # print(line.intersection(polygon_box))

    floor = np.load("floor_2d_path.npy").T

    polygon_box = cp.get_floor_polygon(floor)

    tm = time.time()
    path = visibility_path(start, end, polygon_box)
    # path = np.vstack((path, start))
    print(path)
    # print(dist)
    print("computation time = ", time.time() - tm)

    plot_visibility_path(polygon_box, path)
