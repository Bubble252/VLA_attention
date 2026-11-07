import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_goal13_offline_eval import evaluate_offline_eval


class Goal13OfflineEvalTests(unittest.TestCase):
    def make_fixture(self, root: Path):
        checkpoint = root / "model.safetensors"
        checkpoint.write_bytes(b"checkpoint")
        config = root / "config.yaml"
        config.write_text("config: true\n")
        postrun = root / "postrun.json"
        postrun.write_text(
            json.dumps(
                {
                    "postrun_status": "PASS",
                    "arm": "D0",
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": "a" * 64,
                    "config_snapshot": str(config),
                    "config_snapshot_sha256": "b" * 64,
                }
            )
        )
        offline = root / "offline.json"
        offline.write_text(
            json.dumps(
                {
                    "schema": "goal12-c1-counterfactual-offline-v1",
                    "arm": "D0",
                    "checkpoint": {"path": str(checkpoint), "sha256": "a" * 64},
                    "config": {"path": str(config), "sha256": "b" * 64},
                    "seed": 17,
                    "num_groups": 118,
                    "pair_results": {
                        "destination": {
                            "pair_count": 78,
                            "correct_mae": 1.0,
                            "margin_mean": 0.1,
                            "action_amplitude_ratio": 1.1,
                            "translation_amplitude_ratio": 1.2,
                            "rotation_amplitude_ratio": 1.3,
                            "gripper_amplitude_ratio": 1.4,
                        },
                        "object": {
                            "pair_count": 78,
                            "correct_mae": 1.0,
                            "margin_mean": 0.2,
                            "action_amplitude_ratio": 1.1,
                            "translation_amplitude_ratio": 1.2,
                            "rotation_amplitude_ratio": 1.3,
                            "gripper_amplitude_ratio": 1.4,
                        },
                    },
                    "region_results": None,
                }
            )
        )
        return postrun, offline

    def evaluate(self, postrun: Path, offline: Path):
        return evaluate_offline_eval(
            postrun_path=postrun,
            offline_path=offline,
            expected_groups=118,
            expected_destination_pairs=78,
            expected_object_pairs=78,
        )

    def test_valid_full_eval_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            postrun, offline = self.make_fixture(Path(directory))
            report = self.evaluate(postrun, offline)
        self.assertEqual(report["contract_status"], "PASS")
        self.assertEqual(report["gate_status"], "DESCRIPTIVE")
        self.assertEqual(report["failures"], [])

    def test_wrong_group_count_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            postrun, offline = self.make_fixture(Path(directory))
            payload = json.loads(offline.read_text())
            payload["num_groups"] = 2
            offline.write_text(json.dumps(payload))
            report = self.evaluate(postrun, offline)
        self.assertEqual(report["contract_status"], "INVALID")
        self.assertIn("num_groups", report["failures"])

    def test_wrong_pair_count_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            postrun, offline = self.make_fixture(Path(directory))
            payload = json.loads(offline.read_text())
            payload["pair_results"]["destination"]["pair_count"] = 2
            offline.write_text(json.dumps(payload))
            report = self.evaluate(postrun, offline)
        self.assertEqual(report["contract_status"], "INVALID")
        self.assertIn("destination_pair_count", report["failures"])

    def test_checkpoint_sha_mismatch_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            postrun, offline = self.make_fixture(Path(directory))
            payload = json.loads(offline.read_text())
            payload["checkpoint"]["sha256"] = "c" * 64
            offline.write_text(json.dumps(payload))
            report = self.evaluate(postrun, offline)
        self.assertEqual(report["contract_status"], "INVALID")
        self.assertIn("checkpoint_sha256", report["failures"])

    def test_nonfinite_metric_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            postrun, offline = self.make_fixture(Path(directory))
            payload = json.loads(offline.read_text())
            payload["pair_results"]["destination"]["correct_mae"] = float("nan")
            offline.write_text(json.dumps(payload))
            report = self.evaluate(postrun, offline)
        self.assertEqual(report["contract_status"], "INVALID")
        self.assertIn("all_metrics_finite", report["failures"])


if __name__ == "__main__":
    unittest.main()
