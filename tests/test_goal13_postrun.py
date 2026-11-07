import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_goal13_postrun import evaluate_postrun


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class Goal13PostrunTests(unittest.TestCase):
    def make_fixture(self, root: Path):
        run_dir = root / "run"
        run_dir.mkdir()
        snapshot = root / "goal13_d0_snapshot.yaml"
        snapshot.write_text("snapshot: true\n")
        snapshot_sha = sha256_bytes(snapshot.read_bytes())
        config_full = run_dir / "config.full.yaml"
        config_full.write_text(
            "\n".join(
                [
                    "seed: 17",
                    f"output_dir: {run_dir}",
                    f"config_yaml: {snapshot}",
                    "datasets:",
                    "  vla_data:",
                    "    per_device_batch_size: 4",
                    "trainer:",
                    "  max_train_steps: 2",
                    "  gradient_accumulation_steps: 1",
                    "",
                ]
            )
        )
        summary = run_dir / "summary.jsonl"
        summary.write_text('{"steps": 2}\n')
        train_log = root / "train.log"
        train_log.write_text(
            "\n".join(
                [
                    "Dropped incomplete action chunks; effective dataset lengths: [11630]",
                    "Dataset lengths: [11630]",
                    "Total optimization steps = 2",
                    "Per device batch size = 4",
                    "Gradient accumulation steps = 1",
                    "Total batch size  train_starvla.py:619",
                    "= 4",
                    "STAR105_RUNTIME_BATCH_CONTRACT accum=1 total_batch=4",
                    "STAR105_RUNTIME_DATASET_CONTRACT libero_90_gate_lerobot:length=11630:weight=1.000000",
                    f"STAR105_RUNTIME_CONFIG_SHA {snapshot_sha}",
                    "Training complete.",
                    "Final model saved at final_model",
                    "",
                ]
            )
        )
        stats = run_dir / "dataset_statistics.json"
        stats.write_text(
            json.dumps(
                {
                    "franka": {
                        "num_transitions": 12302,
                        "num_trajectories": 96,
                    }
                }
            )
        )
        final_checkpoint = run_dir / "final_model" / "model.safetensors"
        final_checkpoint.parent.mkdir()
        final_checkpoint.write_bytes(b"checkpoint-bytes")
        step_checkpoint = run_dir / "checkpoints" / "steps_2_model.safetensors"
        step_checkpoint.parent.mkdir()
        step_checkpoint.write_bytes(b"checkpoint-bytes")
        health = root / "checkpoint_health.json"
        health.write_text(
            json.dumps(
                {
                    "checkpoint": str(final_checkpoint),
                    "sha256": sha256_bytes(final_checkpoint.read_bytes()),
                    "all_finite": True,
                    "nonfinite_tensors": [],
                    "tensor_count": 3,
                }
            )
        )
        contract = root / "D0_contract.json"
        contract.write_text(
            json.dumps(
                {
                    "contract_status": "PASS",
                    "arm": "D0",
                    "marker_mode": "none",
                    "counterfactual_mode": "none",
                    "code_commit": "b" * 40,
                    "config_sha256": snapshot_sha,
                    "runtime": {
                        "per_device_batch": 4,
                        "num_processes": 1,
                        "grad_accum": 1,
                        "effective_batch": 4,
                        "seed": 17,
                    },
                    "data": {
                        "dataset_lengths": {"libero_90_gate_lerobot": 11630},
                        "sampling_weights": {"libero_90_gate_lerobot": 1.0},
                    },
                }
            )
        )
        return {
            "contract": contract,
            "run_dir": run_dir,
            "train_log": train_log,
            "snapshot": snapshot,
            "health": health,
        }

    def test_matched_smoke_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            report = evaluate_postrun(
                contract_path=fixture["contract"],
                run_dir=fixture["run_dir"],
                train_log=fixture["train_log"],
                config_snapshot=fixture["snapshot"],
                checkpoint_health_path=fixture["health"],
                expected_steps=2,
            )
        self.assertEqual(report["postrun_status"], "PASS")
        self.assertEqual(report["arm_status"], "SMOKE_PASS")
        self.assertEqual(report["failures"], [])

    def test_runtime_markers_required(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            report = evaluate_postrun(
                contract_path=fixture["contract"],
                run_dir=fixture["run_dir"],
                train_log=fixture["train_log"],
                config_snapshot=fixture["snapshot"],
                checkpoint_health_path=fixture["health"],
                expected_steps=2,
                require_runtime_markers=True,
            )
        self.assertEqual(report["postrun_status"], "PASS")
        self.assertTrue(report["checks"]["log_runtime_batch_marker"]["passed"])
        self.assertTrue(report["checks"]["log_runtime_dataset_marker"]["passed"])
        self.assertTrue(report["checks"]["log_runtime_config_sha"]["passed"])

    def test_nonfinite_checkpoint_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            health = json.loads(fixture["health"].read_text())
            health["all_finite"] = False
            health["nonfinite_tensors"] = ["action_head.weight"]
            fixture["health"].write_text(json.dumps(health))
            report = evaluate_postrun(
                contract_path=fixture["contract"],
                run_dir=fixture["run_dir"],
                train_log=fixture["train_log"],
                config_snapshot=fixture["snapshot"],
                checkpoint_health_path=fixture["health"],
                expected_steps=2,
            )
        self.assertEqual(report["postrun_status"], "INVALID")
        self.assertIn("checkpoint_health_all_finite", report["failures"])
        self.assertIn("checkpoint_health_nonfinite_empty", report["failures"])

    def test_wrong_summary_step_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            (fixture["run_dir"] / "summary.jsonl").write_text('{"steps": 1}\n')
            report = evaluate_postrun(
                contract_path=fixture["contract"],
                run_dir=fixture["run_dir"],
                train_log=fixture["train_log"],
                config_snapshot=fixture["snapshot"],
                checkpoint_health_path=fixture["health"],
                expected_steps=2,
            )
        self.assertEqual(report["postrun_status"], "INVALID")
        self.assertIn("summary_steps", report["failures"])

    def test_checkpoint_copy_mismatch_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            (fixture["run_dir"] / "checkpoints" / "steps_2_model.safetensors").write_bytes(
                b"different-bytes"
            )
            report = evaluate_postrun(
                contract_path=fixture["contract"],
                run_dir=fixture["run_dir"],
                train_log=fixture["train_log"],
                config_snapshot=fixture["snapshot"],
                checkpoint_health_path=fixture["health"],
                expected_steps=2,
            )
        self.assertEqual(report["postrun_status"], "INVALID")
        self.assertIn("checkpoint_copy_identical", report["failures"])

    def test_expected_predicted_marker_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            contract = json.loads(fixture["contract"].read_text())
            contract["arm"] = "I2"
            contract["marker_mode"] = "predicted"
            fixture["contract"].write_text(json.dumps(contract))
            report = evaluate_postrun(
                contract_path=fixture["contract"],
                run_dir=fixture["run_dir"],
                train_log=fixture["train_log"],
                config_snapshot=fixture["snapshot"],
                checkpoint_health_path=fixture["health"],
                expected_steps=2,
                expected_arm="I2",
                expected_marker_mode="predicted",
            )
        self.assertEqual(report["postrun_status"], "PASS")
        self.assertEqual(report["marker_mode"], "predicted")

    def test_wrong_marker_mode_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            report = evaluate_postrun(
                contract_path=fixture["contract"],
                run_dir=fixture["run_dir"],
                train_log=fixture["train_log"],
                config_snapshot=fixture["snapshot"],
                checkpoint_health_path=fixture["health"],
                expected_steps=2,
                expected_marker_mode="predicted",
            )
        self.assertEqual(report["postrun_status"], "INVALID")
        self.assertIn("contract_marker_mode", report["failures"])


if __name__ == "__main__":
    unittest.main()
