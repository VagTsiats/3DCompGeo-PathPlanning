import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import *
import cv2


def get_floor_polygon(grid):

    # Convert the occupancy grid to a binary image
    # Assume 1 is occupied, so we want to keep those and set everything else to 0
    binary_grid = (grid == 1).astype(np.uint8) * 255

    # cv2.CHAIN_APPROX_SIMPLE only necessary points of contour
    #
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


if __name__ == "__main__":

    # Plot the original occupancy grid and the polygon image
    plt.figure(figsize=(12, 6))

    grid = np.load("floor_2d_path.npy")

    plt.subplot(1, 2, 1)
    plt.title("Original Occupancy Grid")
    plt.imshow(grid, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Obstacle Borders with Polygons")
    plt.imshow(grid, cmap="gray")
    plt.axis("off")

    polygon = get_floor_polygon(grid)

    # polygon = create_polygon_with_holes(floor_grid, object_grid)

    x, y = polygon.exterior.xy
    plt.plot(x, y, color="red", label="Main Polygon")

    for poly in polygon.interiors:
        x, y = poly.xy
        plt.plot(x, y, color="blue")

    plt.show()
