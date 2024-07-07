import numpy as np
import alphashape

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


def object_polygon(grid, alpha):

    occupied_cells = np.argwhere(grid == 1)

    try:
        polygon = alphashape.alphashape(occupied_cells, alpha)

        shape = np.array(polygon.exterior.coords, int)
    except:
        return None

    polygon = remove_collinear_points(shape)

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

    # Define the occupancy grid
    grid = np.zeros((20, 20), dtype=int)
    grid[5:15, 8:12] = 1  # Obstacle
    grid[9:11, 2:25] = 1
    # grid[7:13, 4:18] = 1

    grid = np.load("object2dmask.npy")

    poly = object_polygon(grid, 0.2)
    visualize(poly)
