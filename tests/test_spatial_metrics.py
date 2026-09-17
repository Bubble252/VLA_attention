import unittest

try:
    import numpy as np
except ImportError:
    np = None

from vla_attention.evaluation.spatial import mass_in_boxes, pointing_correct


@unittest.skipIf(np is None, "local system Python lacks a working NumPy binary; execute numerical metric test in P1 env")
class SpatialMetricTests(unittest.TestCase):
    def test_peak_and_mass_use_image_coordinates(self):
        grid = np.array([[0.0, 0.0], [0.0, 2.0]])
        box = [[50, 50, 100, 100]]
        self.assertTrue(pointing_correct(grid, box, image_width=100, image_height=100))
        self.assertEqual(mass_in_boxes(grid, box, image_width=100, image_height=100), 1.0)

    def test_overlapping_boxes_do_not_double_count_mass(self):
        grid = np.ones((2, 2))
        boxes = [[0, 0, 100, 100], [25, 25, 75, 75]]
        self.assertEqual(mass_in_boxes(grid, boxes, image_width=100, image_height=100), 1.0)
