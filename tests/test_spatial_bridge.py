import unittest

from vla_attention.spatial import bilinear_weights


class SpatialBridgeTests(unittest.TestCase):
    def test_weights_sum_to_one_and_cover_requested_grid(self):
        rows = bilinear_weights(16, 16, 17, 18)
        self.assertEqual(len(rows), 17 * 18)
        for row in rows:
            self.assertAlmostEqual(sum(weight for _, weight in row), 1.0)
            self.assertTrue(all(0 <= index < 256 for index, _ in row))

    def test_identity_grid_is_identity(self):
        rows = bilinear_weights(2, 2, 2, 2)
        self.assertEqual(rows, [[(0, 1.0)], [(1, 1.0)], [(2, 1.0)], [(3, 1.0)]])
