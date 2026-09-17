import unittest

from scripts.run_sd_ddim_map import find_subsequence


class SDMapTokenTests(unittest.TestCase):
    def test_finds_phrase_span_inside_caption_tokens(self):
        self.assertEqual(find_subsequence([1, 4, 9, 8, 2], [9, 8]), [2, 3])

    def test_rejects_missing_phrase_span(self):
        with self.assertRaises(ValueError):
            find_subsequence([1, 2, 3], [4])
