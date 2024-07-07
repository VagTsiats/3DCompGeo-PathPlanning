import numpy as np
from shapely.geometry import *
from shapely.ops import unary_union
import matplotlib.pyplot as plt

from time import time

DISTANCE_TOLERANCE = 1e-6


def get_next_vert(i, polygon, direction=1):
    i = i + direction

    if i > len(polygon.exterior.coords) - 2:
        i = 0
    elif i < 0:
        i = len(polygon.exterior.coords) - 2

    return i, polygon.exterior.coords[i]


def get_closest_intersection(p1, p2, polygon: Polygon):
    p1p2 = LineString([p1, p2])

    intersections = p1p2.intersection(polygon)

    # Start point of the line
    start_point = Point(p1)
    end_point = Point(p2)

    # Extract intersection points
    intersection_points = []

    for intersection in intersections:
        if isinstance(intersection, Point):
            intersection_points.append(intersection)
        elif isinstance(intersection, MultiPoint):
            intersection_points.extend(intersection.geoms)
        elif isinstance(intersection, LineString):
            intersection_points.extend([Point(coord) for coord in intersection.coords])
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


def visibility_path(p1, p2, polygons: Polygon):

    path = np.array([p1])
    distance = 0

    closest_intersection = get_closest_intersection(p1, p2, polygons)

    # print(closest_intersection)

    if closest_intersection != None:
        # find intersecting polygon
        for polygon in polygons:
            if polygon.distance(closest_intersection) < DISTANCE_TOLERANCE:
                break

        # find intersecting edge of polygon
        for i in range(len(polygon.exterior.coords) - 1):
            edge = LineString([polygon.exterior.coords[i], polygon.exterior.coords[i + 1]])

            # a = edge.line_locate_point(closest_intersection)

            if edge.distance(closest_intersection) < DISTANCE_TOLERANCE:
                icw, vcw = get_next_vert(i, polygon, 1)
                iccw = i
                vccw = polygon.exterior.coords[iccw]
                vcw_path, _ = visibility_path(p1, vcw, polygons)
                vccw_path, _ = visibility_path(p1, vccw, polygons)
                pathcw = np.vstack((path, vcw_path))
                pathccw = np.vstack((path, vccw_path))
                break

        vcwp2 = LineString([vcw, p2])
        vccwp2 = LineString([vccw, p2])

        while vcwp2.crosses(polygon):
            icw, vcw = get_next_vert(icw, polygon, 1)
            vcwp2 = LineString([vcw, p2])
            pathcw = np.vstack((pathcw, vcw))

            # for p in reversed(path):
            # print(p)
            # p = p1
            # p1v = LineString([p, vcw])
            # if not p1v.crosses(polygon):
            #     p1vpath, p1vdist = visibility_path(p, vcw, polygons)
            #     if p1vdist < total_distance(pathcw):
            #         pathcw = p1vpath

        while vccwp2.crosses(polygon):
            iccw, vccw = get_next_vert(iccw, polygon, -1)
            vccwp2 = LineString([vccw, p2])
            pathccw = np.vstack((pathccw, vccw))

            # p = p1
            # p1vccw = LineString([p, vccw])
            # if not p1vccw.crosses(polygon):
            #     p1vpath, p1vdist = visibility_path(p, vccw, polygons)
            #     if p1vdist < total_distance(pathccw):
            #         pathccw = p1vpath

        path_cw, dist_cw = visibility_path(vcw, p2, polygons)
        path_ccw, dist_ccw = visibility_path(vccw, p2, polygons)

        path_cw = np.delete(path_cw, 0, axis=0)
        path_ccw = np.delete(path_ccw, 0, axis=0)

        pathcw = np.vstack((pathcw, path_cw))
        pathccw = np.vstack((pathccw, path_ccw))

        if total_distance(pathcw) <= total_distance(pathccw):
            path = pathcw
        else:
            path = pathccw

        # path = pathccw

    else:
        path = np.vstack((path, p2))

    distance = total_distance(path)

    return path.astype(float), distance


def plot_visibility_path(polygons, path):
    fig, ax = plt.subplots()

    # Plot polygons
    for poly in polygons:
        x, y = poly.exterior.xy
        ax.plot(x, y, color="blue")
        for interior in poly.interiors:  # Plot holes
            ix, iy = zip(*interior.coords)
            ax.plot(ix, iy, color="blue")

    # Plot visibility edges
    for i in range(len(path) - 1):
        x_values = [path[i][0], path[i + 1][0]]
        y_values = [path[i][1], path[i + 1][1]]
        ax.plot(x_values, y_values, color="red", linestyle="--")

    ax.set_aspect("equal", "box")
    plt.show()


if __name__ == "__main__":
    polygons = []
    path = []

    start = (7, 0)
    end = (80, 60)

    shapes = np.load("dataset/myfiles/room_polys.npz")

    for poly in shapes:
        polygon = shapes[poly]

        polygons.append(Polygon(polygon))

    merged_polygon = unary_union(polygons)

    polygons = []

    for poly in merged_polygon.geoms:
        polygons.append(poly)

    # polygon1 = Polygon([(1, 1), (5, 1), (10, 5), (5, 4), (2, 3), (1, 4), (1, 1)])
    # # polygon2 = Polygon([(6, 2), (9, 2), (9, 4.5), (6, 4.5), (6, 2)])
    # # polygon3 = Polygon([(2, 5), (4, 5), (4, 7), (2, 7), (2, 5)])

    # polygons = [polygon1]

    # start = (0, 2.5)
    # end = (8, 2.5)

    tm = time()
    path, dist = visibility_path(start, end, polygons)
    print(path)
    print(dist)
    print("computation time = ", time() - tm)

    plot_visibility_path(polygons, path)
