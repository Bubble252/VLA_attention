#!/usr/bin/env python3
"""Fail-closed preflight for Goal13 I-layer training configs.

This script does not import torch. It checks that the YAML actually expresses
the registered I0/I2 intervention before any GPU work starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


SCHEMA = "goal13-i-config-preflight-v1"


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping in {path}")
    return payload


def add(checks: dict[str, Any], failures: list[str], name: str, passed: bool, expected: Any, actual: Any) -> None:
    checks[name] = {"passed": bool(passed), "expected": expected, "actual": actual}
    if not passed:
        failures.append(name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(
    *,
    config_path: Path,
    arm: str,
    init_checkpoint: Path,
    point_checkpoint: Path | None,
    trainable_modules: list[str],
    point_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    framework = config.get("framework", {}) if isinstance(config.get("framework"), dict) else {}
    trainer = config.get("trainer", {}) if isinstance(config.get("trainer"), dict) else {}
    dataset = (
        config.get("datasets", {}).get("vla_data", {})
        if isinstance(config.get("datasets"), dict)
        else {}
    )
    point = framework.get("point_condition", {}) if isinstance(framework.get("point_condition"), dict) else {}
    lr = trainer.get("learning_rate", {}) if isinstance(trainer.get("learning_rate"), dict) else {}
    freeze = str(trainer.get("freeze_modules", ""))
    checks: dict[str, Any] = {}
    failures: list[str] = []

    add(checks, failures, "arm_supported", arm in {"I0", "I2"}, ["I0", "I2"], arm)
    add(checks, failures, "seed", int(config.get("seed", -1)) == 17, 17, config.get("seed"))
    add(checks, failures, "max_train_steps_positive", int(trainer.get("max_train_steps", 0)) > 0, ">0", trainer.get("max_train_steps"))
    add(checks, failures, "per_device_batch", int(dataset.get("per_device_batch_size", -1)) == 4, 4, dataset.get("per_device_batch_size"))
    add(checks, failures, "grad_accum", int(trainer.get("gradient_accumulation_steps", -1)) == 1, 1, trainer.get("gradient_accumulation_steps"))
    add(checks, failures, "data_mix", dataset.get("data_mix") == "libero_c1_gate", "libero_c1_gate", dataset.get("data_mix"))
    add(
        checks,
        failures,
        "init_checkpoint",
        Path(str(trainer.get("pretrained_checkpoint", ""))).resolve() == init_checkpoint.resolve(),
        str(init_checkpoint.resolve()),
        trainer.get("pretrained_checkpoint"),
    )
    add(checks, failures, "init_checkpoint_exists", init_checkpoint.is_file(), True, init_checkpoint.is_file())
    add(checks, failures, "qwen_frozen", "qwen_vl_interface" in freeze, True, freeze)
    add(checks, failures, "qwen_lr_zero", float(lr.get("qwen_vl_interface", -1.0)) == 0.0, 0.0, lr.get("qwen_vl_interface"))
    add(checks, failures, "action_lr_positive", float(lr.get("action_model", 0.0)) > 0.0, ">0", lr.get("action_model"))

    if arm == "I0":
        add(checks, failures, "i0_marker_disabled", not bool(point.get("enabled", False)), False, point.get("enabled", False))
        add(checks, failures, "i0_trainable_action_only", trainable_modules == ["action_model"], ["action_model"], trainable_modules)
        add(checks, failures, "i0_no_projector_lr", float(lr.get("point_evidence_projector", 0.0)) == 0.0, 0.0, lr.get("point_evidence_projector"))
    else:
        add(checks, failures, "i2_marker_enabled", bool(point.get("enabled", False)) is True, True, point.get("enabled", False))
        add(checks, failures, "i2_injection", point.get("injection") == "evidence", "evidence", point.get("injection"))
        add(
            checks,
            failures,
            "i2_point_checkpoint",
            point_checkpoint is not None
            and Path(str(point.get("checkpoint", ""))).resolve() == point_checkpoint.resolve(),
            str(point_checkpoint.resolve()) if point_checkpoint else None,
            point.get("checkpoint"),
        )
        add(
            checks,
            failures,
            "i2_point_checkpoint_exists",
            point_checkpoint is not None and point_checkpoint.is_file(),
            True,
            point_checkpoint.is_file() if point_checkpoint else None,
        )
        actual_point_sha = (
            sha256_file(point_checkpoint)
            if point_checkpoint is not None and point_checkpoint.is_file()
            else None
        )
        add(
            checks,
            failures,
            "i2_point_checkpoint_sha256",
            bool(point_checkpoint_sha256) and actual_point_sha == point_checkpoint_sha256,
            point_checkpoint_sha256,
            actual_point_sha,
        )
        add(checks, failures, "i2_point_head_frozen", "point_head" in freeze, True, freeze)
        add(checks, failures, "i2_tokenizer_frozen", "point_tokenizer" in freeze, True, freeze)
        add(checks, failures, "i2_projector_lr_positive", float(lr.get("point_evidence_projector", 0.0)) > 0.0, ">0", lr.get("point_evidence_projector"))
        add(
            checks,
            failures,
            "i2_trainable_modules",
            trainable_modules == ["action_model", "point_evidence_projector"],
            ["action_model", "point_evidence_projector"],
            trainable_modules,
        )

    status = "PASS" if not failures else "INVALID"
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "arm": arm,
        "config": str(config_path),
        "checks": checks,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=("I0", "I2"), required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--point-checkpoint", type=Path, default=None)
    parser.add_argument("--point-checkpoint-sha256", default=None)
    parser.add_argument("--trainable-modules", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    modules = [item.strip() for item in args.trainable_modules.split(",") if item.strip()]
    report = evaluate(
        config_path=args.config,
        arm=args.arm,
        init_checkpoint=args.init_checkpoint,
        point_checkpoint=args.point_checkpoint,
        trainable_modules=modules,
        point_checkpoint_sha256=args.point_checkpoint_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "failures": report["failures"], "output": str(args.output)}, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
