import unittest

from vla_attention.contracts import AttributionTarget, CoordinateFrame, SpatialGrid, SpatialMap


class ContractTests(unittest.TestCase):
    def test_spatial_map_accepts_valid_phrase_grid(self):
        grid = SpatialGrid(2, 2, CoordinateFrame.PATCH_GRID, 32, 32, (0, 1, 2, 3))
        spatial_map = SpatialMap("m1", grid, "teacher", "red mug", AttributionTarget.PHRASE_SCORE, "sum_to_one")
        spatial_map.validate()

    def test_spatial_grid_rejects_inconsistent_token_count(self):
        grid = SpatialGrid(2, 2, CoordinateFrame.PATCH_GRID, 32, 32, (0, 1, 2))
        with self.assertRaises(ValueError):
            grid.validate()
