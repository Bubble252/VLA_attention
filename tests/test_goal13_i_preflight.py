from __future__ import annotations

import tempfile
import unittest
import hashlib
from pathlib import Path

import yaml

from scripts.preflight_goal13_i_config import evaluate


def make_config(
    root: Path,
    *,
    arm: str,
    init_checkpoint: Path,
    point_checkpoint: Path | None = None,
    evidence_cross_attention: bool = False,
    evidence_token: bool = False,
):
    point = {}
    freeze = "qwen_vl_interface"
    lr = {"base": 1.0e-4, "qwen_vl_interface": 0.0, "action_model": 1.0e-4}
    if arm == "I2":
        point = {
            "enabled": True,
            "injection": "evidence",
            "checkpoint": str(point_checkpoint),
        }
        freeze = "qwen_vl_interface,point_head,point_tokenizer"
        lr["point_head"] = 0.0
        lr["point_tokenizer"] = 0.0
        lr["point_evidence_projector"] = 1.0e-4
    payload = {
        "seed": 17,
        "framework": {
            "point_condition": point,
            "action_model": {
                "evidence_cross_attention": evidence_cross_attention,
                "evidence_token": evidence_token,
            },
        },
        "datasets": {"vla_data": {"data_mix": "libero_c1_gate", "per_device_batch_size": 4}},
        "trainer": {
            "pretrained_checkpoint": str(init_checkpoint),
            "max_train_steps": 500,
            "gradient_accumulation_steps": 1,
            "freeze_modules": freeze,
            "learning_rate": lr,
        },
    }
    path = root / f"{arm}.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


class Goal13IPreflightTests(unittest.TestCase):
    def test_i0_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            init = root / "d0.safetensors"
            init.write_bytes(b"d0")
            config = make_config(root, arm="I0", init_checkpoint=init)
            report = evaluate(
                config_path=config,
                arm="I0",
                init_checkpoint=init,
                point_checkpoint=None,
                trainable_modules=["action_model"],
            )
        self.assertEqual(report["status"], "PASS", report["failures"])

    def test_i2_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            init = root / "d0.safetensors"
            point = root / "point_head.pt"
            init.write_bytes(b"d0")
            point.write_bytes(b"point")
            point_sha = hashlib.sha256(point.read_bytes()).hexdigest()
            config = make_config(root, arm="I2", init_checkpoint=init, point_checkpoint=point)
            report = evaluate(
                config_path=config,
                arm="I2",
                init_checkpoint=init,
                point_checkpoint=point,
                trainable_modules=["action_model", "point_evidence_projector"],
                point_checkpoint_sha256=point_sha,
            )
        self.assertEqual(report["status"], "PASS", report["failures"])

    def test_i2_with_wrong_point_sha_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            init = root / "d0.safetensors"
            point = root / "point_head.pt"
            init.write_bytes(b"d0")
            point.write_bytes(b"point")
            config = make_config(root, arm="I2", init_checkpoint=init, point_checkpoint=point)
            report = evaluate(
                config_path=config,
                arm="I2",
                init_checkpoint=init,
                point_checkpoint=point,
                trainable_modules=["action_model", "point_evidence_projector"],
                point_checkpoint_sha256="0" * 64,
            )
        self.assertEqual(report["status"], "INVALID")
        self.assertIn("i2_point_checkpoint_sha256", report["failures"])

    def test_i2_with_disabled_marker_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            init = root / "d0.safetensors"
            point = root / "point_head.pt"
            init.write_bytes(b"d0")
            point.write_bytes(b"point")
            config = make_config(root, arm="I2", init_checkpoint=init, point_checkpoint=point)
            payload = yaml.safe_load(config.read_text())
            payload["framework"]["point_condition"]["enabled"] = False
            config.write_text(yaml.safe_dump(payload))
            report = evaluate(
                config_path=config,
                arm="I2",
                init_checkpoint=init,
                point_checkpoint=point,
                trainable_modules=["action_model", "point_evidence_projector"],
            )
        self.assertEqual(report["status"], "INVALID")
        self.assertIn("i2_marker_enabled", report["failures"])

    def test_i2_cross_attention_preflight_requires_enabled_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            init = root / "d0.safetensors"
            point = root / "point_head.pt"
            init.write_bytes(b"d0")
            point.write_bytes(b"point")
            point_sha = hashlib.sha256(point.read_bytes()).hexdigest()
            config = make_config(
                root,
                arm="I2",
                init_checkpoint=init,
                point_checkpoint=point,
                evidence_cross_attention=True,
            )
            report = evaluate(
                config_path=config,
                arm="I2",
                init_checkpoint=init,
                point_checkpoint=point,
                trainable_modules=["action_model", "point_evidence_projector"],
                point_checkpoint_sha256=point_sha,
                expected_evidence_cross_attention=True,
            )
            self.assertEqual(report["status"], "PASS", report["failures"])

            payload = yaml.safe_load(config.read_text())
            payload["framework"]["action_model"]["evidence_cross_attention"] = False
            config.write_text(yaml.safe_dump(payload))
            report = evaluate(
                config_path=config,
                arm="I2",
                init_checkpoint=init,
                point_checkpoint=point,
                trainable_modules=["action_model", "point_evidence_projector"],
                point_checkpoint_sha256=point_sha,
                expected_evidence_cross_attention=True,
            )
        self.assertEqual(report["status"], "INVALID")
        self.assertIn("i2_evidence_cross_attention", report["failures"])

    def test_i2_evidence_token_preflight_requires_enabled_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            init = root / "d0.safetensors"
            point = root / "point_head.pt"
            init.write_bytes(b"d0")
            point.write_bytes(b"point")
            point_sha = hashlib.sha256(point.read_bytes()).hexdigest()
            config = make_config(
                root,
                arm="I2",
                init_checkpoint=init,
                point_checkpoint=point,
                evidence_cross_attention=False,
                evidence_token=True,
            )
            report = evaluate(
                config_path=config,
                arm="I2",
                init_checkpoint=init,
                point_checkpoint=point,
                trainable_modules=["action_model", "point_evidence_projector"],
                point_checkpoint_sha256=point_sha,
                expected_evidence_cross_attention=False,
                expected_evidence_token=True,
            )
            self.assertEqual(report["status"], "PASS", report["failures"])

            payload = yaml.safe_load(config.read_text())
            payload["framework"]["action_model"]["evidence_token"] = False
            config.write_text(yaml.safe_dump(payload))
            report = evaluate(
                config_path=config,
                arm="I2",
                init_checkpoint=init,
                point_checkpoint=point,
                trainable_modules=["action_model", "point_evidence_projector"],
                point_checkpoint_sha256=point_sha,
                expected_evidence_cross_attention=False,
                expected_evidence_token=True,
            )
        self.assertEqual(report["status"], "INVALID")
        self.assertIn("i2_evidence_token", report["failures"])


if __name__ == "__main__":
    unittest.main()
