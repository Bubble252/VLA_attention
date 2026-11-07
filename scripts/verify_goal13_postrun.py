#!/usr/bin/env python3
"""Verify that a Goal13 smoke run matches its pre-registered Run Contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


SCHEMA = "goal13-postrun-contract-v1"
FATAL_PATTERNS = (
    r"Traceback \(most recent call last\)",
    r"CUDA out of memory",
    r"\bNaN\b",
    r"non[- ]finite",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping in {path}")
    return payload


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_jsonl_steps(path: Path) -> int | None:
    steps: list[int] = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict) and "steps" in payload:
            steps.append(int(payload["steps"]))
    return max(steps) if steps else None


def first_int(pattern: str, text: str, flags: int = 0) -> int | None:
    match = re.search(pattern, text, flags=flags)
    return int(match.group(1)) if match else None


def first_int_list(pattern: str, text: str, flags: int = 0) -> list[int] | None:
    match = re.search(pattern, text, flags=flags)
    if not match:
        return None
    return [int(item) for item in re.findall(r"\d+", match.group(1))]


def git_head(repo: Path) -> str | None:
    if not repo.exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def add_check(
    checks: dict[str, dict[str, Any]],
    name: str,
    passed: bool,
    *,
    expected: Any,
    actual: Any,
    detail: str = "",
) -> None:
    checks[name] = {
        "passed": bool(passed),
        "expected": expected,
        "actual": actual,
        "detail": detail,
    }


def evaluate_postrun(
    *,
    contract_path: Path,
    run_dir: Path,
    train_log: Path,
    config_snapshot: Path,
    checkpoint_health_path: Path,
    expected_steps: int,
    repo: Path | None = None,
) -> dict[str, Any]:
    contract = load_json(contract_path)
    checks: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    def check(name: str, passed: bool, expected: Any, actual: Any, detail: str = "") -> None:
        add_check(checks, name, passed, expected=expected, actual=actual, detail=detail)
        if not passed:
            failures.append(name)

    check("contract_status", contract.get("contract_status") == "PASS", "PASS", contract.get("contract_status"))
    check("contract_arm", contract.get("arm") == "D0", "D0", contract.get("arm"))
    check("contract_marker_none", contract.get("marker_mode") == "none", "none", contract.get("marker_mode"))
    check(
        "contract_counterfactual_none",
        contract.get("counterfactual_mode") == "none",
        "none",
        contract.get("counterfactual_mode"),
    )

    expected_code_commit = contract.get("code_commit")
    actual_code_commit = git_head(repo) if repo is not None else expected_code_commit
    check(
        "code_commit",
        bool(actual_code_commit) and actual_code_commit == expected_code_commit,
        expected_code_commit,
        actual_code_commit,
    )

    snapshot_sha = sha256_file(config_snapshot) if config_snapshot.is_file() else None
    check(
        "config_snapshot_sha256",
        bool(snapshot_sha) and snapshot_sha == contract.get("config_sha256"),
        contract.get("config_sha256"),
        snapshot_sha,
    )

    full_config_path = run_dir / "config.full.yaml"
    full_config = load_yaml(full_config_path) if full_config_path.is_file() else {}
    trainer = full_config.get("trainer", {}) if isinstance(full_config.get("trainer"), dict) else {}
    datasets = full_config.get("datasets", {}) if isinstance(full_config.get("datasets"), dict) else {}
    vla_data = datasets.get("vla_data", {}) if isinstance(datasets.get("vla_data"), dict) else {}
    runtime = contract.get("runtime", {}) if isinstance(contract.get("runtime"), dict) else {}
    expected_effective_batch = int(runtime.get("effective_batch", -1))

    check(
        "config_max_train_steps",
        int(trainer.get("max_train_steps", -1)) == expected_steps,
        expected_steps,
        trainer.get("max_train_steps"),
    )
    check(
        "config_per_device_batch",
        int(vla_data.get("per_device_batch_size", -1)) == int(runtime.get("per_device_batch", -2)),
        runtime.get("per_device_batch"),
        vla_data.get("per_device_batch_size"),
    )
    check(
        "config_grad_accum",
        int(trainer.get("gradient_accumulation_steps", -1)) == int(runtime.get("grad_accum", -2)),
        runtime.get("grad_accum"),
        trainer.get("gradient_accumulation_steps"),
    )
    check(
        "config_effective_batch",
        int(vla_data.get("per_device_batch_size", 0))
        * int(trainer.get("gradient_accumulation_steps", 0))
        == expected_effective_batch,
        expected_effective_batch,
        int(vla_data.get("per_device_batch_size", 0))
        * int(trainer.get("gradient_accumulation_steps", 0)),
    )
    check(
        "config_seed",
        int(full_config.get("seed", -1)) == int(runtime.get("seed", -2)),
        runtime.get("seed"),
        full_config.get("seed"),
    )
    check(
        "config_output_dir",
        Path(str(full_config.get("output_dir", ""))).resolve() == run_dir.resolve(),
        str(run_dir.resolve()),
        full_config.get("output_dir"),
    )
    config_yaml = full_config.get("config_yaml")
    check(
        "config_yaml_snapshot",
        bool(config_yaml) and Path(str(config_yaml)).resolve() == config_snapshot.resolve(),
        str(config_snapshot.resolve()),
        config_yaml,
    )

    summary_path = run_dir / "summary.jsonl"
    summary_steps = parse_jsonl_steps(summary_path) if summary_path.is_file() else None
    check("summary_steps", summary_steps == expected_steps, expected_steps, summary_steps)

    log_text = train_log.read_text(errors="replace") if train_log.is_file() else ""
    log_steps = first_int(r"optimization steps\s*=\s*(\d+)", log_text)
    log_per_device_batch = first_int(r"Per device batch[\s\S]{0,100}?size\s*=\s*(\d+)", log_text)
    log_grad_accum = first_int(r"Gradient[\s\S]{0,100}?accumulation steps\s*=\s*(\d+)", log_text)
    log_total_batch = first_int(r"Total batch size\s*=\s*(\d+)", log_text)
    effective_lengths = first_int_list(r"effective dataset lengths:\s*\[([0-9,\s]+)\]", log_text)
    dataset_lengths = first_int_list(r"Dataset lengths:\s*\[([0-9,\s]+)\]", log_text)
    expected_lengths = list(contract.get("data", {}).get("dataset_lengths", {}).values())

    check("log_optimization_steps", log_steps == expected_steps, expected_steps, log_steps)
    check(
        "log_per_device_batch",
        log_per_device_batch == int(runtime.get("per_device_batch", -2)),
        runtime.get("per_device_batch"),
        log_per_device_batch,
    )
    check(
        "log_grad_accum",
        log_grad_accum == int(runtime.get("grad_accum", -2)),
        runtime.get("grad_accum"),
        log_grad_accum,
    )
    check(
        "log_effective_batch",
        log_total_batch == expected_effective_batch,
        expected_effective_batch,
        log_total_batch,
    )
    check(
        "log_dataset_lengths",
        dataset_lengths == expected_lengths and effective_lengths == expected_lengths,
        expected_lengths,
        {"raw": dataset_lengths, "effective": effective_lengths},
    )
    fatal_hits = [pattern for pattern in FATAL_PATTERNS if re.search(pattern, log_text, flags=re.IGNORECASE)]
    check("log_no_fatal_errors", not fatal_hits, [], fatal_hits)
    check(
        "log_training_complete",
        "Training complete" in log_text and "Final model saved at" in log_text,
        True,
        {
            "training_complete": "Training complete" in log_text,
            "final_model_saved": "Final model saved at" in log_text,
        },
    )

    dataset_stats_path = run_dir / "dataset_statistics.json"
    dataset_stats = load_json(dataset_stats_path) if dataset_stats_path.is_file() else {}
    franka = dataset_stats.get("franka", {}) if isinstance(dataset_stats.get("franka"), dict) else {}
    check(
        "dataset_statistics_transitions",
        int(franka.get("num_transitions", -1)) == 12302,
        12302,
        franka.get("num_transitions"),
    )
    check(
        "dataset_statistics_trajectories",
        int(franka.get("num_trajectories", -1)) == 96,
        96,
        franka.get("num_trajectories"),
    )

    final_checkpoint = run_dir / "final_model" / "model.safetensors"
    step_checkpoint = run_dir / "checkpoints" / f"steps_{expected_steps}_model.safetensors"
    check("final_checkpoint_exists", final_checkpoint.is_file(), True, final_checkpoint.is_file())
    check("step_checkpoint_exists", step_checkpoint.is_file(), True, step_checkpoint.is_file())
    final_sha = sha256_file(final_checkpoint) if final_checkpoint.is_file() else None
    step_sha = sha256_file(step_checkpoint) if step_checkpoint.is_file() else None
    check("checkpoint_copy_identical", bool(final_sha) and final_sha == step_sha, final_sha, step_sha)

    health = load_json(checkpoint_health_path) if checkpoint_health_path.is_file() else {}
    check("checkpoint_health_present", checkpoint_health_path.is_file(), True, checkpoint_health_path.is_file())
    check(
        "checkpoint_health_path",
        Path(str(health.get("checkpoint", ""))).resolve() == final_checkpoint.resolve(),
        str(final_checkpoint.resolve()),
        health.get("checkpoint"),
    )
    check("checkpoint_health_sha256", health.get("sha256") == final_sha, final_sha, health.get("sha256"))
    check("checkpoint_health_all_finite", health.get("all_finite") is True, True, health.get("all_finite"))
    check(
        "checkpoint_health_nonfinite_empty",
        health.get("nonfinite_tensors") == [],
        [],
        health.get("nonfinite_tensors"),
    )
    check(
        "checkpoint_health_tensor_count",
        int(health.get("tensor_count", 0)) > 0,
        ">0",
        health.get("tensor_count"),
    )

    postrun_status = "PASS" if not failures else "INVALID"
    return {
        "schema": SCHEMA,
        "created_at": utc_now(),
        "postrun_status": postrun_status,
        "arm_status": "SMOKE_PASS" if postrun_status == "PASS" else "INVALID",
        "gate_status": "NOT_EVALUATED",
        "arm": contract.get("arm"),
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "expected_steps": expected_steps,
        "config_snapshot": str(config_snapshot),
        "config_snapshot_sha256": snapshot_sha,
        "config_full_sha256": sha256_file(full_config_path) if full_config_path.is_file() else None,
        "train_log": str(train_log),
        "train_log_sha256": sha256_file(train_log) if train_log.is_file() else None,
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path) if summary_path.is_file() else None,
        "checkpoint": str(final_checkpoint),
        "checkpoint_sha256": final_sha,
        "checkpoint_size_bytes": final_checkpoint.stat().st_size if final_checkpoint.is_file() else None,
        "checkpoint_health": health,
        "loss_logged": False,
        "finite_evidence": "checkpoint_health",
        "checks": checks,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--config-snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint-health", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=2)
    parser.add_argument("--repo", type=Path, default=None)
    args = parser.parse_args()

    report = evaluate_postrun(
        contract_path=args.contract,
        run_dir=args.run_dir,
        train_log=args.train_log,
        config_snapshot=args.config_snapshot,
        checkpoint_health_path=args.checkpoint_health,
        expected_steps=args.expected_steps,
        repo=args.repo,
    )
    atomic_json_write(args.output, report)
    print(
        json.dumps(
            {
                "postrun_status": report["postrun_status"],
                "arm_status": report["arm_status"],
                "failures": report["failures"],
                "checkpoint_sha256": report["checkpoint_sha256"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if report["postrun_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
