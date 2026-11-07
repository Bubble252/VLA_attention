from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from scripts.audit_goal13_rlds import PROMPTS, iter_tfrecord_payloads, sha256_bytes
from scripts.extract_goal13_carrier import CARRIER_PROMPT, extract_carrier, load_carrier_rows


def write_tfrecord(path: Path, payloads: list[bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for payload in payloads:
            handle.write(struct.pack("<Q", len(payload)))
            handle.write(b"\x00\x00\x00\x00")
            handle.write(payload)
            handle.write(b"\x00\x00\x00\x00")


class Goal13CarrierExtractTests(unittest.TestCase):
    def make_fixture(self, root: Path, *, count: int = 3):
        shard_a = root / "data" / "part-00000.tfrecord-00000-of-00002"
        shard_b = root / "data" / "part-00001.tfrecord-00001-of-00002"
        language = PROMPTS[CARRIER_PROMPT]
        shard_a_payloads = [b"not-carrier", b"a0-" + language, b"a2-" + language]
        shard_b_payloads = [b"b0-" + language, b"not-carrier-2"]
        write_tfrecord(shard_a, shard_a_payloads)
        write_tfrecord(shard_b, shard_b_payloads)

        selected = [
            (shard_b, 0, shard_b_payloads[0]),
            (shard_a, 2, shard_a_payloads[2]),
            (shard_a, 1, shard_a_payloads[1]),
        ][:count]
        rows = [
            {
                "episode_id": sha256_bytes(payload),
                "language": language.decode("utf-8"),
                "prompt": CARRIER_PROMPT,
                "source_shard": str(shard),
                "source_record_index": index,
                "source_shard_index": 0,
                "split": "unassigned",
            }
            for shard, index, payload in selected
        ]
        audit = root / "audit.json"
        audit.write_text(json.dumps({"episodes": rows + [{"prompt": "red_left"}]}))
        split = root / "split.json"
        split.write_text(
            json.dumps(
                {
                    "splits": {
                        "train": [{"episode_id": "train-id"}],
                        "validation": [],
                        "rollout": [],
                    }
                }
            )
        )
        return audit, split, selected

    def test_extracts_exact_ordered_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit, split, selected = self.make_fixture(root)
            report = extract_carrier(
                audit_path=audit,
                output_tfrecord=root / "out.tfrecord",
                output_manifest=root / "manifest.jsonl",
                output_report=root / "report.json",
                source_root=root,
                split_manifest_path=split,
                expected_count=3,
            )
            payloads = list(iter_tfrecord_payloads(root / "out.tfrecord"))
            manifest = [
                json.loads(line)
                for line in (root / "manifest.jsonl").read_text().splitlines()
                if line
            ]
        expected = [payload for _, _, payload in sorted(selected, key=lambda row: (str(row[0]), row[1]))]
        self.assertEqual(payloads, expected)
        self.assertEqual([row["output_record_index"] for row in manifest], [0, 1, 2])
        self.assertEqual(report["split_overlap_total"], 0)
        self.assertEqual(report["carrier_count"], 3)

    def test_carrier_split_overlap_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit, split, selected = self.make_fixture(root)
            payload = json.loads(split.read_text())
            payload["splits"]["train"].append({"episode_id": sha256_bytes(selected[0][2])})
            split.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "overlap C1 splits"):
                extract_carrier(
                    audit_path=audit,
                    output_tfrecord=root / "out.tfrecord",
                    output_manifest=root / "manifest.jsonl",
                    output_report=root / "report.json",
                    source_root=root,
                    split_manifest_path=split,
                    expected_count=3,
                )

    def test_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit, split, _ = self.make_fixture(root)
            payload = json.loads(audit.read_text())
            payload["episodes"][0]["episode_id"] = "0" * 64
            audit.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                extract_carrier(
                    audit_path=audit,
                    output_tfrecord=root / "out.tfrecord",
                    output_manifest=root / "manifest.jsonl",
                    output_report=root / "report.json",
                    source_root=root,
                    split_manifest_path=split,
                    expected_count=3,
                )

    def test_count_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            audit, _, _ = self.make_fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "carrier count mismatch"):
                load_carrier_rows(audit, expected_count=48)


if __name__ == "__main__":
    unittest.main()
