import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_goal13_offline_smoke import evaluate_offline_smoke


class Goal13OfflineSmokeTests(unittest.TestCase):
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
                    "num_groups": 2,
                    "pair_results": {
                        "destination": {
                            "pair_count": 2,
                            "correct_mae": 1.0,
                            "margin_mean": 0.1,
                            "action_amplitude_ratio": 1.1,
                            "translation_amplitude_ratio": 1.2,
                            "rotation_amplitude_ratio": 1.3,
                            "gripper_amplitude_ratio": 1.4,
                        },
                        "object": {
                            "pair_count": 2,
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

    def test_valid_smoke_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            postrun, offline = self.make_fixture(Path(directory))
            report = evaluate_offline_smoke(
                postrun_path=postrun,
                offline_path=offline,
                expected_limit=2,
            )
        self.assertEqual(report["postrun_status"], "SMOKE_PASS")
        self.assertEqual(report["gate_status"], "NOT_EVALUATED")
        self.assertEqual(report["failures"], [])

    def test_valid_d1_smoke_passes_with_expected_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            postrun, offline = self.make_fixture(Path(directory))
            payload = json.loads(offline.read_text())
            payload["arm"] = "D1"
            offline.write_text(json.dumps(payload))
            report = evaluate_offline_smoke(
                postrun_path=postrun,
                offline_path=offline,
                expected_limit=2,
                expected_arm="D1",
            )
        self.assertEqual(report["postrun_status"], "SMOKE_PASS")
        self.assertEqual(report["failures"], [])

    def test_arm_mismatch_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            postrun, offline = self.make_fixture(Path(directory))
            payload = json.loads(offline.read_text())
            payload["arm"] = "D1"
            offline.write_text(json.dumps(payload))
            report = evaluate_offline_smoke(
                postrun_path=postrun,
                offline_path=offline,
                expected_limit=2,
                expected_arm="D0",
            )
        self.assertEqual(report["postrun_status"], "INVALID")
        self.assertIn("offline_arm", report["failures"])

    def test_checkpoint_sha_mismatch_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            postrun, offline = self.make_fixture(Path(directory))
            payload = json.loads(offline.read_text())
            payload["checkpoint"]["sha256"] = "c" * 64
            offline.write_text(json.dumps(payload))
            report = evaluate_offline_smoke(
                postrun_path=postrun,
                offline_path=offline,
                expected_limit=2,
            )
        self.assertEqual(report["postrun_status"], "INVALID")
        self.assertIn("checkpoint_sha256", report["failures"])

    def test_nonfinite_metric_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            postrun, offline = self.make_fixture(Path(directory))
            payload = json.loads(offline.read_text())
            payload["pair_results"]["destination"]["correct_mae"] = float("nan")
            offline.write_text(json.dumps(payload))
            report = evaluate_offline_smoke(
                postrun_path=postrun,
                offline_path=offline,
                expected_limit=2,
            )
        self.assertEqual(report["postrun_status"], "INVALID")
        self.assertIn("all_metrics_finite", report["failures"])

    def test_wrong_group_count_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            postrun, offline = self.make_fixture(Path(directory))
            payload = json.loads(offline.read_text())
            payload["num_groups"] = 3
            offline.write_text(json.dumps(payload))
            report = evaluate_offline_smoke(
                postrun_path=postrun,
                offline_path=offline,
                expected_limit=2,
            )
        self.assertEqual(report["postrun_status"], "INVALID")
        self.assertIn("num_groups", report["failures"])


if __name__ == "__main__":
    unittest.main()
