import unittest

from vla_attention.adapters.qwen25 import QwenImageGrid


class QwenGridTests(unittest.TestCase):
    def test_restores_post_merge_non_square_grid(self):
        grid = QwenImageGrid(1, 12, 20, 2)
        spatial = grid.spatial_grid(image_height=336, image_width=560, image_token_indices=range(60))
        self.assertEqual((spatial.height, spatial.width), (6, 10))
        self.assertEqual(spatial.metadata["token_stage"], "post_spatial_merge")
        spatial.validate()

    def test_rejects_video_flattening_and_wrong_token_count(self):
        with self.assertRaises(ValueError):
            QwenImageGrid(2, 12, 12, 2).spatial_grid(image_height=336, image_width=336, image_token_indices=range(72))
        with self.assertRaises(ValueError):
            QwenImageGrid(1, 12, 12, 2).spatial_grid(image_height=336, image_width=336, image_token_indices=range(35))
