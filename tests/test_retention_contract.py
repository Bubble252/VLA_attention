import unittest

from vla_attention.spatial import bilinear_weights


class RetentionContractTests(unittest.TestCase):
    def test_dino_to_qwen_bridge_is_explicit_not_token_index_based(self):
        bridge = bilinear_weights(16, 16, 17, 18)
        self.assertEqual(len(bridge), 306)
        self.assertTrue(any(len(row) > 1 for row in bridge))
