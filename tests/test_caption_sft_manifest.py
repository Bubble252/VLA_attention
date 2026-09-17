import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import build_caption_sft_manifest as script


class CaptionManifestTests(unittest.TestCase):
    def test_uses_only_entities_train_intersection_and_all_captions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); entities, captions, output = root / "e.jsonl", root / "c.csv", root / "out.jsonl"
            entities.write_text(json.dumps({"image_id": "a", "image_path": "images/a.jpg"}) + "\n")
            with captions.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["raw", "filename"]); writer.writeheader()
                writer.writerow({"filename": "a.jpg", "raw": json.dumps(["first caption", "second caption"])})
                writer.writerow({"filename": "b.jpg", "raw": json.dumps(["excluded"])})
            with patch("sys.argv", ["tool", "--entities-train", str(entities), "--captions-csv", str(captions), "--output", str(output)]): script.main()
            rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual([row["caption"] for row in rows], ["first caption", "second caption"])
        self.assertTrue(all(row["prompt"] == script.PROMPT for row in rows))
