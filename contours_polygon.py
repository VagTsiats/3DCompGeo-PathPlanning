import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import *
import cv2


def get_floor_polygon(grid):

    # Convert the occupancy grid to a binary image
    # Assume 1 is occupied, so we want to keep those and set everything else to 0
    binary_grid = (grid == 1).astype(np.uint8) * 255

    # Find contours and hierarchy using OpenCV
    contours, hierarchy = cv2.findContours(binary_grid, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize outer and inner polygons
    main_polygon = None
    holes = []

    # Convert contours to polygons using shapely
    for i, contour in enumerate(contours):
        if contour.shape[0] > 2:  # Ensure it has enough points to form a polygon
            polygon = Polygon(contour.squeeze())

            # Check if it is an outer or inner contour based on hierarchy
            if hierarchy[0][i][3] == -1:  # No parent means outer polygon
                main_polygon = polygon
            else:  # Has a parent means inner polygon
                holes.append(polygon)

    if main_polygon:
        shapely_polygon = Polygon(main_polygon.exterior.coords, [hole.exterior.coords for hole in holes])

    shapely_polygon = shapely_polygon.buffer(0)

    # If the resulting geometry is a MultiPolygon, take the largest one
    if shapely_polygon.type == "MultiPolygon":
        shapely_polygon = max(shapely_polygon.geoms, key=lambda p: p.area)

    return shapely_polygon


# def is_collinear(p1, p2, p3):
#     """Check if three points are collinear."""
#     return (p3.y - p1.y) * (p2.x - p1.x) == (p2.y - p1.y) * (p3.x - p1.x)


# def remove_collinear_points(polygon):
#     """Remove collinear points from a polygon."""
#     if not isinstance(polygon, Polygon):
#         raise TypeError("Input must be a Shapely Polygon")

#     ring = LinearRing(polygon.exterior.coords)
#     non_collinear_coords = []

#     for i in range(len(ring.coords) - 2):
#         p1 = Point(ring.coords[i])
#         p2 = Point(ring.coords[i + 1])
#         p3 = Point(ring.coords[i + 2])

#         if not is_collinear(p1, p2, p3):
#             non_collinear_coords.append(ring.coords[i + 1])

#     # Adding the first point and the last point
#     non_collinear_coords.insert(0, ring.coords[0])
#     non_collinear_coords.append(ring.coords[-1])

#     return Polygon(non_collinear_coords)


if __name__ == "__main__":

    # Plot the original occupancy grid and the polygon image
    plt.figure(figsize=(12, 6))

    grid = np.load("floor_2d_path.npy").T

    plt.subplot(1, 2, 1)
    plt.title("Original Occupancy Grid")
    plt.imshow(grid, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Obstacle Borders with Polygons")
    plt.imshow(grid, cmap="gray")
    plt.axis("off")

    polygon = get_floor_polygon(grid)

    print(len(polygon.exterior.coords))

    x, y = polygon.exterior.xy
    plt.plot(x, y, color="red", label="Main Polygon")

    for poly in polygon.interiors:
        x, y = poly.xy
        plt.plot(x, y, color="blue")

    plt.show()
