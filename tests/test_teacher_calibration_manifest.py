import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import build_teacher_calibration_manifest as script


class TeacherCalibrationManifestTests(unittest.TestCase):
    def test_one_deterministic_row_per_image(self):
        rows = [
            {"image_id": "b", "phrase": "zebra", "sample_id": "b2", "boxes_xyxy": [[1, 1, 2, 2]]},
            {"image_id": "b", "phrase": "apple", "sample_id": "b1", "boxes_xyxy": [[1, 1, 2, 2]]},
            {"image_id": "a", "phrase": "dog", "sample_id": "a1", "boxes_xyxy": [[1, 1, 2, 2]]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "val.jsonl", root / "cal.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with patch("sys.argv", ["tool", "--entities-val", str(source), "--output", str(output)]):
                script.main()
            selected = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual([row["sample_id"] for row in selected], ["a1", "b1"])
