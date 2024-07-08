import numpy as np
from shapely.geometry import *
from shapely.ops import unary_union
import matplotlib.pyplot as plt

from time import time

DISTANCE_TOLERANCE = 1e-6

fig, ax = plt.subplots()


def get_next_vert(i, polygon, direction=1):
    i = i + direction

    if i > len(polygon.exterior.coords) - 2:
        i = 0
    elif i < 0:
        i = len(polygon.exterior.coords) - 2

    return i, polygon.exterior.coords[i]


def get_closest_intersection(p1, p2, polygons):
    p1p2 = LineString([p1, p2])

    intersections = p1p2.intersection(polygons)

    # print(intersections)

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


def visibility_path(p1, p2, polygons: Polygon, d=0):
    # print("")
    # print("visipath", p1, p2, d)

    path = np.array([p1])

    closest_intersection = get_closest_intersection(p1, p2, polygons)

    if d == 10:
        closest_intersection = None

    # print("closest intersection :", closest_intersection)

    for poly in polygons:
        if poly.intersects(Point(p1)) or poly.intersects(Point(p2)):
            polygon = poly
            break

    if closest_intersection != None:
        # find intersecting polygon
        for poly in polygons:
            if poly.distance(closest_intersection) < DISTANCE_TOLERANCE:
                polygon = poly
                break

        # find intersecting edge of polygon
        for i in range(len(polygon.exterior.coords) - 1):
            edge = LineString([polygon.exterior.coords[i], polygon.exterior.coords[i + 1]])

            if p1 == polygon.exterior.coords[i]:  # and (p2 != polygon.exterior.coords[i + 1] or p2 != polygon.exterior.coords[i - 1]):
                icw, vcw = get_next_vert(i, polygon, 1)
                iccw, vccw = get_next_vert(i, polygon, -1)
                break
            elif edge.distance(closest_intersection) < DISTANCE_TOLERANCE:
                icw, vcw = get_next_vert(i, polygon, 1)
                iccw = i
                vccw = polygon.exterior.coords[iccw]
                break

        p1_vcw_path = visibility_path(p1, vcw, polygons, d + 1)
        p1_vccw_path = visibility_path(p1, vccw, polygons, d + 1)

        vcw_p2_line = LineString([vcw, p2])
        vccw_p2_line = LineString([vccw, p2])

        while vcw_p2_line.crosses(polygon):
            icw, vcw = get_next_vert(icw, polygon, 1)
            p1_vcw_path = np.vstack((p1_vcw_path, vcw))

            # for i, p in enumerate(reversed(p1_vcw_path)) :
            for i in range(len(p1_vcw_path) - 2, -1, -1):
                p_vcw_line = LineString([p1_vcw_path[i], vcw])

                if not polygon.crosses(p_vcw_line) and not polygon.contains(p_vcw_line):
                    p_vcw_path = visibility_path(tuple(p1_vcw_path[i]), vcw, polygons, d + 1)
                    p_vcw_path = np.vstack((p1_vcw_path[:i], p_vcw_path))

                    if total_distance(p_vcw_path) < total_distance(p1_vcw_path):
                        p1_vcw_path = p_vcw_path

            vcw_p2_line = LineString([vcw, p2])

        while vccw_p2_line.crosses(polygon):
            iccw, vccw = get_next_vert(iccw, polygon, -1)
            p1_vccw_path = np.vstack((p1_vccw_path, vccw))

            for i in range(len(p1_vccw_path) - 2, -1, -1):
                p_vccw_line = LineString([p1_vccw_path[i], vccw])

                if not polygon.crosses(p_vccw_line) and not polygon.contains(p_vccw_line):
                    p_vccw_path = visibility_path(tuple(p1_vccw_path[i]), vccw, polygons, d + 1)
                    p_vccw_path = np.vstack((p1_vccw_path[:i], p_vccw_path))

                    if total_distance(p_vccw_path) < total_distance(p1_vccw_path):
                        p1_vccw_path = p_vccw_path

            vccw_p2_line = LineString([vccw, p2])

        vcw_p2_path = visibility_path(vcw, p2, polygons, d + 1)
        vccw_p2_path = visibility_path(vccw, p2, polygons, d + 1)

        vcw_p2_path = np.delete(vcw_p2_path, 0, axis=0)
        vccw_p2_path = np.delete(vccw_p2_path, 0, axis=0)

        p1p2cw_path = np.vstack((p1_vcw_path, vcw_p2_path))
        p1p2ccw_path = np.vstack((p1_vccw_path, vccw_p2_path))

        # print(total_distance(p1p2cw_path), total_distance(p1p2ccw_path))

        if total_distance(p1p2cw_path) <= total_distance(p1p2ccw_path):
            path = p1p2cw_path
        else:
            path = p1p2ccw_path

        # path = p1p2cw_path

    else:
        path = np.vstack((path, p2))

    # print("visipath_end", p1, p2, d)
    # print(total_distance(path))
    # print("")

    # for i in range(len(path) - 1):
    #     x_values = [path[i][0], path[i + 1][0]]
    #     y_values = [path[i][1], path[i + 1][1]]
    #     ax.plot(x_values, y_values, color="red", linestyle="--")

    # ax.set_aspect("equal", "box")
    # plt.show()

    return path.astype(float)


def plot_visibility_path(polygons, path):

    # Plot polygons
    for i, poly in enumerate(polygons):
        x, y = poly.exterior.xy
        if i == 0:
            ax.plot(x, y, color="red")
        else:
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

    end = (240, 80)
    start = (210, 130)

    # shapes = np.load("dataset/myfiles/room_polys.npz")
    shapes = np.load("room_polys.npz")

    for poly in shapes:
        polygon = shapes[poly]

        polygons.append(Polygon(polygon))

    merged_polygon = unary_union(polygons)

    floor_poly = np.load("floor_poly.npy", allow_pickle=True)

    polygons = []

    polygons.append(Polygon(floor_poly))

    for poly in merged_polygon.geoms:
        polygons.append(poly)

    result = polygons[0]

    for poly in polygons[1:]:
        result = result.difference(poly)

    if isinstance(result, GeometryCollection):
        # Extract only Polygon geometries from the GeometryCollection
        result = [geom for geom in result.geoms if isinstance(geom, Polygon)]

        # If there's more than one resulting polygon, you may want to merge them
        # if result:
        #     result = unary_union(result)
        # else:
        #     result = None

    # print(result)

    for poly in polygons[:1]:
        result.append(poly)

    polygons = result

    print(polygons)

    # polygon_box = Polygon([(0, 0), (4, 0), (5, 6), (6, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
    # polygon1 = Polygon([(1, 1), (3, 4.5), (1, 4), (1, 1)])
    # polygon2 = Polygon([(6, 2), (8, 2), (7, 5)])

    # polygons = [polygon_box, polygon1, polygon2]

    # start = (0.5, 1)
    # end = (8, 1)

    # line = LineString([start, end])
    # p = Point((5, 0))

    # print(line.intersection(polygon_box))
    # print(p.intersects(polygon1))
    # print(polygon1.crosses(line))

    # plot_visibility_path(polygons, path)

    # cliscks = plt.ginput(1)

    # print(cliscks)

    tm = time()
    # path = visibility_path(start, end, polygons)
    # path = np.vstack((path, start))
    # print(path)
    # print(dist)
    print("computation time = ", time() - tm)

    plot_visibility_path([polygons[0]], path)
