import unittest

from vla_attention.teachers.null_text import NullTextResult


class NullTextTests(unittest.TestCase):
    def test_result_keeps_trajectory_and_loss_provenance(self):
        result = NullTextResult([1, 2], [3], [0.1])
        self.assertEqual(result.inverse_latents[-1], 2)
        self.assertEqual(result.unconditional_embeddings, [3])
        self.assertEqual(result.reconstruction_losses, [0.1])
