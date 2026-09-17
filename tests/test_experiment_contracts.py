import json
import tempfile
import unittest
from pathlib import Path

from vla_attention.audit_io import read_report
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

    def test_audit_report_requires_declared_target_and_valid_grid(self):
        payload = {
            "capability": {
                "model_id": "qwen2.5-vl-7b", "model_revision": "test",
                "supports_visual_tokens": True, "supports_hidden_states": True,
                "supported_targets": ["phrase_score"], "coordinate_notes": [], "limitations": [],
            },
            "measurements": [{
                "sample_id": "s1", "model_id": "qwen2.5-vl-7b", "target": "phrase_score",
                "map_path": "/vepfs/example.npy", "seed": 17, "scalar_definition": "phrase logprob",
                "gradient_finite": True, "repeatability": 0.98,
                "grid": {"height": 2, "width": 2, "frame": "patch_grid", "image_height": 32,
                         "image_width": 32, "token_indices": [0, 1, 2, 3]},
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(payload))
            capability, rows = read_report(path)
        self.assertEqual(capability.model_id, "qwen2.5-vl-7b")
        self.assertEqual(len(rows), 1)
