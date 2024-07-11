import visibility_path
from shapely.geometry import *
import numpy as np
import unittest as ut
import matplotlib.pylab as plt


class Path_test(ut.TestCase):

    def test_start_point_polygon_vertex_end_point_after_object(self):
        polygon1 = Polygon([(1, 1), (5, 1), (5.5, 5), (5, 4), (2, 3), (1, 10), (1, 1)])
        polygon2 = Polygon([(6, 2), (9, 2), (9, 4.5), (8, 5), (6, 4.5), (6, 2)])
        polygon3 = Polygon([(2, 5), (4, 5), (4, 7), (2, 7), (2, 5)])

        polygons = [polygon1, polygon2, polygon3]

        start = (1, 1)
        end = (8.5, 5)

        path = visibility_path.visibility_path(start, end, polygons)

        correct_path = np.array([[1, 1.0], [5.0, 1.0], [6.0, 4.5], [8.0, 5.0], [8.5, 5.0]])

        self.assertTrue(np.array_equal(path, correct_path))

    def test_start_point_polygon_vertex_end_point_other_object_vertex(self):
        polygon1 = Polygon([(1, 1), (5, 1), (5.5, 5), (5, 4), (2, 3), (1, 10), (1, 1)])
        polygon2 = Polygon([(6, 2), (9, 2), (9, 4.5), (8, 5), (6, 4.5), (6, 2)])
        polygon3 = Polygon([(2, 5), (4, 5), (4, 7), (2, 7), (2, 5)])

        polygons = [polygon1, polygon2, polygon3]

        start = (1, 1)
        end = (8, 5)

        path = visibility_path.visibility_path(start, end, polygons)

        correct_path = np.array([[1, 1.0], [5.0, 1.0], [6.0, 4.5], [8.0, 5.0]])

        self.assertTrue(np.array_equal(path, correct_path))

    # unoccuring event
    # def test_start_point_polygon_vertex_end_point_other_vertex(self):
    #     polygon1 = Polygon([(1, 1), (5, 1), (5.5, 5), (5, 4), (2, 3), (1, 10), (1, 1)])
    #     polygon2 = Polygon([(6, 2), (9, 2), (9, 4.5), (8, 5), (6, 4.5), (6, 2)])
    #     polygon3 = Polygon([(2, 5), (4, 5), (4, 7), (2, 7), (2, 5)])

    #     polygons = [polygon1, polygon2, polygon3]

    #     start = (1, 1)
    #     end = (5, 4)

    #     path = visibility_path.visibility_path(start, end, polygons)

    #     correct_path = np.array([[1.0, 1.0], [5.0, 1.0], [5.5, 5], [5, 4]])

    #     self.assertTrue(np.array_equal(path, correct_path))

    def test_start_point_polygon_vertex_end_point_other_vertex_colinear(self):
        polygon1 = Polygon([(1, 1), (3, 1), (5, 1), (5.5, 5), (5, 4), (2, 3), (1, 10), (1, 1)])
        polygon2 = Polygon([(6, 2), (9, 2), (9, 4.5), (8, 5), (6, 4.5), (6, 2)])
        polygon3 = Polygon([(2, 5), (4, 5), (4, 7), (2, 7), (2, 5)])

        polygon1 = visibility_path.remove_collinear_points(polygon1)

        polygons = [polygon1, polygon2, polygon3]

        start = (1, 1)
        end = (5, 1)

        path = visibility_path.visibility_path(start, end, polygons)

        correct_path = np.array([[1.0, 1.0], [5.0, 1.0]])

        self.assertTrue(np.array_equal(path, correct_path))


class Path_inside_map(ut.TestCase):

    def test_inside_box(self):
        # inside box
        polygon_box = Polygon([(0, 0), (4, 0), (4, 6), (6, 5), (6, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        polygon1 = Polygon([(1, 1), (3, 4.5), (1, 4), (1, 1)])
        polygon2 = Polygon([(6.5, 2), (8, 2), (7, 5)])

        polygons = [polygon_box, polygon1, polygon2]

        start = (0.5, 1)
        end = (8, 1)

        path = visibility_path.visibility_path(start, end, polygons)
        correct_path = np.array([[0.5, 1.0], [1.0, 1.0], [4.0, 6.0], [6.0, 5.0], [6.5, 2.0], [8.0, 1.0]])

        self.assertTrue(np.array_equal(path, correct_path))


if __name__ == "__main__":
    ut.main()

    print("PASSES ALL")
