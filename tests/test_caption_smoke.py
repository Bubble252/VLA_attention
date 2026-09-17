import unittest

from scripts.run_qwen_caption_smoke import last_subsequence


class CaptionSmokeTests(unittest.TestCase):
    def test_uses_last_caption_occurrence(self):
        self.assertEqual(last_subsequence([1, 4, 2, 4, 2], [4, 2]), 3)
    def test_rejects_missing_caption(self):
        with self.assertRaises(ValueError): last_subsequence([1, 2], [3])
