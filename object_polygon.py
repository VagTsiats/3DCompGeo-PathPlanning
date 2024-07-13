import numpy as np
import alphashape
from scipy.ndimage import sobel
import cv2

import matplotlib.pyplot as plt


def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if np.isclose(val, 0):
        return 0  # Collinear
    elif val > 0:
        return 1  # Clockwise
    else:
        return 2  # Counterclockwise


def remove_collinear_points(polygon):
    if len(polygon) < 3:
        return polygon

    cleaned_polygon = [polygon[0]]
    n = len(polygon)
    i = 1

    while i < n:
        while i < n - 1 and orientation(cleaned_polygon[-1], polygon[i], polygon[i + 1]) == 0:
            i += 1
        cleaned_polygon.append(polygon[i])
        i += 1

    # Check if the last point is collinear with the first two points
    if len(cleaned_polygon) > 2 and orientation(cleaned_polygon[-3], cleaned_polygon[-2], cleaned_polygon[-1]) == 0:
        cleaned_polygon.pop()

    return np.array(cleaned_polygon)


def object_polygon(grid, alpha=None):

    occupied_cells = np.argwhere(grid == 1)

    try:
        polygon = alphashape.alphashape(occupied_cells, alpha)

        polygon = np.array(polygon.exterior.coords, int)
    except:
        return None

    polygon = remove_collinear_points(polygon)

    return polygon


def visualize(polygon):
    plt.figure(figsize=(10, 6))

    # Visualize
    plt.imshow(grid, cmap="Greys", origin="lower")
    plt.plot(polygon[:, 1], polygon[:, 0], "b-", label="Resulting Polygon")
    plt.scatter(polygon[:, 1], polygon[:, 0], color="blue", label="Polygon Points")

    plt.title("Polygon with Maximum Edge Length Constraint from Occupancy Grid")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    grid = np.load("object2dmask.npy")

    grid = grid[:][150:]

    binary_grid = (grid == 1).astype(np.uint8) * 255  # Multiply by 255 to get a proper binary image for OpenCV

    # Find contours
    contours, _ = cv2.findContours(binary_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty image to draw contours
    contour_image = np.zeros_like(binary_grid)

    # Draw contours on the empty image
    cv2.drawContours(contour_image, contours, -1, (255), 1)

    # Plot the original occupancy grid and the contour image
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Occupancy Grid")
    plt.imshow(binary_grid, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Obstacle Borders")
    plt.imshow(contour_image, cmap="gray")
    plt.axis("off")

    plt.show()

    poly = object_polygon(grid)
    visualize(poly)
