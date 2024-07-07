from shapely.geometry import Point, LineString, Polygon
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def is_visible(p1, p2, polygons):
    line = LineString([p1, p2])
    for polygon in polygons:
        # Check if the line intersects the polygon's exterior (not just touching)
        if polygon.exterior.crosses(line):
            return False
        # Check if the line is inside the polygon (i.e., lies completely inside it)
        if polygon.contains(line):
            return False
        # # Check if the line intersects the polygon's exterior (considering non-convexity)
        # if line.crosses(polygon.exterior):
        #     return False
        # # Check if the line is inside the polygon (non-convex polygons included)
        # if polygon.contains(line):
        #     return False

        # # Check if the line touches any polygon edge initially but doesn't cross
        # if any(
        #     line.touches(LineString([polygon.exterior.coords[i], polygon.exterior.coords[i + 1]])) for i in range(len(polygon.exterior.coords) - 1)
        # ):
        #     # We still need to check if it touches and then crosses the polygon
        #     if any(
        #         line.crosses(LineString([polygon.exterior.coords[i], polygon.exterior.coords[i + 1]]))
        #         for i in range(len(polygon.exterior.coords) - 1)
        #     ):
        #         return False

        # Check for holes (interiors) within the polygon
        for interior in polygon.interiors:
            if LineString(interior).crosses(line):
                return False

    return True


def visibility_graph(polygons):
    vertices = []
    for poly in polygons:
        vertices.extend(list(poly.exterior.coords[:-1]))  # Collect vertices, skip repeated last point
    edges = []
    for p1, p2 in tqdm(combinations(vertices, 2), desc="visibi"):
        if is_visible(p1, p2, polygons):
            edges.append((p1, p2))
    return edges


def plot_visibility_graph(polygons, edges):
    fig, ax = plt.subplots()

    # Plot polygons
    for poly in polygons:
        x, y = poly.exterior.xy
        ax.plot(x, y, color="blue")
        for interior in poly.interiors:  # Plot holes
            ix, iy = zip(*interior.coords)
            ax.plot(ix, iy, color="blue")

    # Plot visibility edges
    for p1, p2 in edges:
        x_values = [p1[0], p2[0]]
        y_values = [p1[1], p2[1]]
        ax.plot(x_values, y_values, color="red", linestyle="--")

    ax.set_aspect("equal", "box")
    plt.show()


if __name__ == "__main__":
    shapes = np.load("room_polys.npz")

    polygons = []
    edges = []

    for poly in shapes:
        polygon = shapes[poly]

        polygons.append(Polygon(polygon))

    # # # Example usage with multiple polygons
    polygon1 = Polygon([(1, 1), (5, 1), (5.5, 5), (5, 4), (3, 3), (1, 4), (1, 1)])
    polygon2 = Polygon([(6, 2), (9, 2), (9, 4.5), (6, 4.5), (6, 2)])
    polygon3 = Polygon([(2, 5), (4, 5), (4, 7), (2, 7), (2, 5)])

    polygons = [polygon1, polygon2, polygon3]

    edges = visibility_graph(polygons)
    plot_visibility_graph(polygons, edges)
