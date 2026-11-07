#!/usr/bin/env python3
"""Build a fail-closed Goal13 Run Contract from a Stage 0 data audit.

The builder does not import the training stack.  It consumes the read-only
RLDS audit, the frozen C1 split manifest, and the observed dataset contract
emitted by the training data loader.  Missing or mismatched evidence makes the
contract INVALID rather than silently falling back to YAML guesses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "goal13-run-contract-v1"
SPLITS = ("train", "validation", "rollout")
C1_PROMPTS = ("red_left", "red_right", "white_left")
HELDOUT_PROMPT = "white_right"
CARRIER_PROMPT = "yellow_and_white_right"
WHITE_BOWL_PROMPT = "white_bowl_right"
CARRIER_DATASET_NAME = "libero_90_goal13_carrier_lerobot"
WHITE_BOWL_DATASET_NAME = "libero_90_goal13_white_bowl_lerobot"
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
I_ARM_MARKERS = {
    "I0": "none",
    "I1": "oracle",
    "I2": "predicted",
    "I3": "shuffle",
}
I_ARM_PARENTS = {
    "I0": "D0",
    "I1": "I2",
    "I2": "D0",
    "I3": "I2",
}
I_ARM_MARKER_SOURCES = {
    "I0": "none",
    "I1": "gt_point_manifest",
    "I2": "point_head",
    "I3": "point_head",
}
I_ARM_DIAGNOSTIC = {"I0": False, "I1": True, "I2": False, "I3": True}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object per line in {path}")
            rows.append(payload)
    return rows


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def default_data_arms() -> dict[str, Any]:
    return {
        "schema": "goal13-data-arms-v1",
        "prompts": {
            "red_left": "put the red mug on the left plate",
            "red_right": "put the red mug on the right plate",
            "white_left": "put the white mug on the left plate",
            "white_right": "put the white mug on the right plate",
            "yellow_and_white_right": "put the yellow and white mug on the right plate",
            "white_bowl_right": "put the white bowl to the right of the plate",
        },
        "expected_source_counts": {
            "red_left": 37,
            "red_right": 44,
            "white_left": 45,
            "white_right": 0,
            "yellow_and_white_right": 48,
            "white_bowl_right": 44,
        },
        "expected_train_counts": {"red_left": 27, "red_right": 34, "white_left": 35},
        "expected_manifest_split_counts": {"train": 96, "validation": 15, "rollout": 15},
        "arms": {
            "D0": {
                "include_carrier_train": False,
                "include_white_bowl_train": False,
                "expected_train_episodes": 96,
            },
            "D1": {
                "include_carrier_train": True,
                "include_white_bowl_train": False,
                "expected_train_episodes": 144,
            },
            "D2": {
                "include_carrier_train": True,
                "include_white_bowl_train": True,
                "expected_train_episodes": 188,
            },
        },
    }


def load_data_arms(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return default_data_arms()
    payload = load_json(path)
    base = default_data_arms()
    base.update(payload)
    base.setdefault("arms", {})
    for name, definition in default_data_arms()["arms"].items():
        base["arms"].setdefault(name, definition)
    return base


def manifest_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    ids_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    for split, rows in payload.get("splits", {}).items():
        if split not in ids_by_split:
            raise ValueError(f"unknown split in manifest: {split}")
        for row in rows:
            episode_id = str(row["episode_id"])
            if episode_id in rows_by_id:
                raise ValueError(f"duplicate episode in split manifest: {episode_id}")
            enriched = dict(row)
            enriched["split"] = split
            rows_by_id[episode_id] = enriched
            ids_by_split[split].add(episode_id)
    overlap = {
        f"{left}__{right}": len(ids_by_split[left] & ids_by_split[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    return {
        "rows_by_id": rows_by_id,
        "ids_by_split": ids_by_split,
        "split_counts": {split: len(ids_by_split[split]) for split in SPLITS},
        "language_counts": dict(Counter(str(row.get("language", "")) for row in rows_by_id.values())),
        "overlap": overlap,
        "overlap_total": sum(overlap.values()),
    }


def normalize_observed_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    datasets: dict[str, dict[str, float | int]] = {}
    raw_datasets = payload.get("datasets", [])
    if isinstance(raw_datasets, list):
        for row in raw_datasets:
            name = str(row["name"])
            weight = row.get("sampling_weight", row.get("weight"))
            if weight is None:
                raise ValueError(f"dataset {name!r} has no sampling_weight")
            datasets[name] = {"length": int(row["length"]), "sampling_weight": float(weight)}
    elif isinstance(raw_datasets, dict):
        for name, row in raw_datasets.items():
            weight = row.get("sampling_weight", row.get("weight"))
            if weight is None:
                raise ValueError(f"dataset {name!r} has no sampling_weight")
            datasets[str(name)] = {"length": int(row["length"]), "sampling_weight": float(weight)}
    else:
        raise ValueError("observed contract must contain a datasets list or mapping")
    weight_sum = float(payload.get("sampling_weight_sum", sum(row["sampling_weight"] for row in datasets.values())))
    return {
        "datasets": datasets,
        "dataset_lengths": {name: int(row["length"]) for name, row in datasets.items()},
        "sampling_weights": {name: float(row["sampling_weight"]) for name, row in datasets.items()},
        "sampling_weight_sum": weight_sum,
    }


def carrier_manifest_summary(
    rows: list[Mapping[str, Any]],
    *,
    expected_language: str,
    split_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    ids: list[str] = []
    languages: Counter[str] = Counter()
    for row in rows:
        episode_id = str(row.get("episode_id", ""))
        if not episode_id:
            raise ValueError("carrier manifest row is missing episode_id")
        ids.append(episode_id)
        languages[str(row.get("language", ""))] += 1
    split_ids = {split: set() for split in SPLITS}
    for split, split_rows in split_manifest.get("splits", {}).items():
        if split not in split_ids:
            raise ValueError(f"unknown split in manifest: {split}")
        split_ids[split].update(str(row["episode_id"]) for row in split_rows)
    carrier_ids = set(ids)
    overlap = {split: len(carrier_ids & values) for split, values in split_ids.items()}
    return {
        "count": len(ids),
        "unique_count": len(carrier_ids),
        "ids": sorted(carrier_ids),
        "languages": dict(languages),
        "expected_language": expected_language,
        "split_overlap": overlap,
        "split_overlap_total": sum(overlap.values()),
    }


def _check(
    name: str,
    passed: bool,
    *,
    expected: Any = None,
    actual: Any = None,
    detail: str = "",
) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "expected": expected,
        "actual": actual,
        "detail": detail,
    }


def evaluate_contract(
    *,
    audit: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    data_arm: str,
    arm_definition: Mapping[str, Any],
    runtime: Mapping[str, Any],
    code_commit: str | None,
    config_sha256: str | None,
    observed_contract: Mapping[str, Any] | None,
    parent_arm: str | None,
    marker_mode: str,
    counterfactual_mode: str,
    expected_source_counts: Mapping[str, int],
    expected_train_counts: Mapping[str, int],
    expected_manifest_split_counts: Mapping[str, int],
    expected_carrier_count: int,
    expected_white_bowl_count: int,
    expected_c1_dataset_length: int,
    c1_dataset_name: str,
    expected_carrier_dataset_length: int | None,
    expected_c1_sampling_weight: float | None,
    expected_carrier_sampling_weight: float | None,
    carrier_rows: list[Mapping[str, Any]] | None,
    expected_raw_transitions: int | None,
    expected_trajectories: int | None,
    prompt_text: Mapping[str, str],
    expected_seed: int,
    expected_effective_batch: int,
    arm: str | None = None,
    marker_source: str | None = None,
    init_checkpoint: str | None = None,
    init_checkpoint_sha256: str | None = None,
    point_checkpoint: str | None = None,
    point_checkpoint_sha256: str | None = None,
    trainable_modules: list[str] | None = None,
    diagnostic_only: bool = False,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    def add(
        name: str,
        passed: bool,
        *,
        expected: Any = None,
        actual: Any = None,
        detail: str = "",
    ) -> None:
        checks[name] = _check(name, passed, expected=expected, actual=actual, detail=detail)
        if not passed:
            failures.append(name)

    summary = manifest_summary(split_manifest)
    source_counts = {name: int(audit.get("source_language_counts", {}).get(name, 0)) for name in expected_source_counts}
    audit_train_counts = {
        name: int(audit.get("train_language_counts", {}).get(name, 0))
        for name in (*expected_train_counts, HELDOUT_PROMPT)
    }
    expected_audit_train_counts = dict(expected_train_counts)
    expected_audit_train_counts[HELDOUT_PROMPT] = 0
    audit_split_counts = audit.get("split_language_counts", {})

    add("audit_status", audit.get("audit_status") == "PASS", expected="PASS", actual=audit.get("audit_status"))
    add(
        "audit_source_counts",
        source_counts == dict(expected_source_counts),
        expected=dict(expected_source_counts),
        actual=source_counts,
    )
    add(
        "audit_train_counts",
        audit_train_counts == expected_audit_train_counts,
        expected=expected_audit_train_counts,
        actual=audit_train_counts,
    )
    add(
        "audit_split_overlap_zero",
        int(audit.get("split_overlap_total", -1)) == 0,
        expected=0,
        actual=audit.get("split_overlap_total"),
    )
    add(
        "white_right_train_zero",
        int(audit.get("white_right_train_count", -1)) == 0,
        expected=0,
        actual=audit.get("white_right_train_count"),
    )
    add(
        "carrier_count",
        int(audit.get("carrier_episode_count", -1)) == expected_carrier_count,
        expected=expected_carrier_count,
        actual=audit.get("carrier_episode_count"),
    )
    add(
        "white_bowl_right_count",
        int(audit.get("white_bowl_right_episode_count", -1)) == expected_white_bowl_count,
        expected=expected_white_bowl_count,
        actual=audit.get("white_bowl_right_episode_count"),
    )

    add(
        "manifest_split_counts",
        summary["split_counts"] == dict(expected_manifest_split_counts),
        expected=dict(expected_manifest_split_counts),
        actual=summary["split_counts"],
    )
    expected_manifest_language_counts = {
        prompt_text[name]: int(expected_source_counts[name]) for name in C1_PROMPTS
    }
    add(
        "manifest_language_counts",
        summary["language_counts"] == expected_manifest_language_counts,
        expected=expected_manifest_language_counts,
        actual=summary["language_counts"],
    )
    add(
        "manifest_overlap_zero",
        summary["overlap_total"] == 0,
        expected=0,
        actual=summary["overlap_total"],
        detail=str(summary["overlap"]),
    )

    per_device_batch = int(runtime.get("per_device_batch", 0))
    num_processes = int(runtime.get("num_processes", 0))
    grad_accum = int(runtime.get("grad_accum", 0))
    seed = int(runtime.get("seed", -1))
    effective_batch = per_device_batch * num_processes * grad_accum
    add("effective_batch_positive", per_device_batch > 0 and num_processes > 0 and grad_accum > 0, expected=">0", actual=effective_batch)
    add("effective_batch", effective_batch == expected_effective_batch, expected=expected_effective_batch, actual=effective_batch)
    add("seed", seed == expected_seed, expected=expected_seed, actual=seed)

    add(
        "code_commit",
        bool(code_commit and COMMIT_PATTERN.fullmatch(code_commit)),
        expected="40-character git commit",
        actual=code_commit,
    )
    add(
        "config_sha256",
        bool(config_sha256 and SHA256_PATTERN.fullmatch(config_sha256)),
        expected="64-character config SHA256",
        actual=config_sha256,
    )

    normalized_observed: dict[str, Any] | None = None
    if observed_contract is None:
        add(
            "observed_dataset_contract",
            False,
            expected="provided by the actual training data loader",
            actual=None,
            detail="run star105_dump_dataset_contract.py against the exact launch config",
        )
    else:
        try:
            normalized_observed = normalize_observed_contract(observed_contract)
            datasets = normalized_observed["datasets"]
            add("observed_dataset_present", bool(datasets), expected=">=1 dataset", actual=sorted(datasets))
            add(
                "observed_sampling_weight_sum",
                abs(float(normalized_observed["sampling_weight_sum"]) - 1.0) <= 0.01,
                expected="1.0 +/- 0.01",
                actual=normalized_observed["sampling_weight_sum"],
            )
            add(
                "observed_lengths_positive",
                all(int(row["length"]) > 0 for row in datasets.values()),
                expected="all lengths > 0",
                actual=normalized_observed["dataset_lengths"],
            )
            c1_row = datasets.get(c1_dataset_name)
            add(
                "c1_dataset_present",
                c1_row is not None,
                expected=c1_dataset_name,
                actual=sorted(datasets),
            )
            if c1_row is not None:
                add(
                    "c1_dataset_length",
                    int(c1_row["length"]) == expected_c1_dataset_length,
                    expected=expected_c1_dataset_length,
                    actual=int(c1_row["length"]),
                )
            if data_arm in {"D1", "D2"}:
                add(
                    "carrier_dataset_present",
                    CARRIER_DATASET_NAME in datasets,
                    expected=CARRIER_DATASET_NAME,
                    actual=sorted(datasets),
                )
                carrier_row = datasets.get(CARRIER_DATASET_NAME)
                if carrier_row is not None and expected_carrier_dataset_length is not None:
                    add(
                        "carrier_dataset_length",
                        int(carrier_row["length"]) == expected_carrier_dataset_length,
                        expected=expected_carrier_dataset_length,
                        actual=int(carrier_row["length"]),
                    )
                if expected_c1_sampling_weight is not None:
                    add(
                        "c1_sampling_weight",
                        c1_row is not None
                        and abs(float(c1_row["sampling_weight"]) - expected_c1_sampling_weight) <= 0.005,
                        expected=expected_c1_sampling_weight,
                        actual=float(c1_row["sampling_weight"]) if c1_row is not None else None,
                    )
                if carrier_row is not None and expected_carrier_sampling_weight is not None:
                    add(
                        "carrier_sampling_weight",
                        abs(float(carrier_row["sampling_weight"]) - expected_carrier_sampling_weight) <= 0.005,
                        expected=expected_carrier_sampling_weight,
                        actual=float(carrier_row["sampling_weight"]),
                    )
                allowed = {c1_dataset_name, CARRIER_DATASET_NAME}
                if data_arm == "D2":
                    allowed.add(WHITE_BOWL_DATASET_NAME)
                add(
                    "observed_dataset_names",
                    set(datasets) == allowed,
                    expected=sorted(allowed),
                    actual=sorted(datasets),
                )
                if carrier_rows is None:
                    add(
                        "carrier_manifest",
                        False,
                        expected="ordered carrier episode manifest",
                        actual=None,
                    )
                else:
                    carrier_summary = carrier_manifest_summary(
                        carrier_rows,
                        expected_language=prompt_text[CARRIER_PROMPT],
                        split_manifest=split_manifest,
                    )
                    add(
                        "carrier_manifest_count",
                        carrier_summary["count"] == expected_carrier_count,
                        expected=expected_carrier_count,
                        actual=carrier_summary["count"],
                    )
                    add(
                        "carrier_manifest_ids_unique",
                        carrier_summary["unique_count"] == carrier_summary["count"],
                        expected=carrier_summary["count"],
                        actual=carrier_summary["unique_count"],
                    )
                    add(
                        "carrier_manifest_language",
                        carrier_summary["languages"]
                        == {prompt_text[CARRIER_PROMPT]: expected_carrier_count},
                        expected={prompt_text[CARRIER_PROMPT]: expected_carrier_count},
                        actual=carrier_summary["languages"],
                    )
                    add(
                        "carrier_manifest_split_overlap_zero",
                        carrier_summary["split_overlap_total"] == 0,
                        expected=0,
                        actual=carrier_summary["split_overlap"],
                    )
        except Exception as error:
            add("observed_dataset_contract", False, expected="valid observed contract", actual=str(error))

    train_language_counts = dict(audit_train_counts)
    if bool(arm_definition.get("include_carrier_train")):
        train_language_counts[CARRIER_PROMPT] = source_counts.get(CARRIER_PROMPT, 0)
    if bool(arm_definition.get("include_white_bowl_train")):
        train_language_counts[WHITE_BOWL_PROMPT] = source_counts.get(WHITE_BOWL_PROMPT, 0)
    train_episode_count = sum(train_language_counts.values())
    expected_train_episodes = int(arm_definition.get("expected_train_episodes", -1))
    add(
        "arm_train_episode_count",
        train_episode_count == expected_train_episodes,
        expected=expected_train_episodes,
        actual=train_episode_count,
    )
    add(
        "arm_white_right_train_zero",
        int(train_language_counts.get(HELDOUT_PROMPT, 0)) == 0,
        expected=0,
        actual=int(train_language_counts.get(HELDOUT_PROMPT, 0)),
    )
    add(
        "heldout_prompt_train_count",
        int(train_language_counts.get(HELDOUT_PROMPT, 0)) == 0,
        expected=0,
        actual=int(train_language_counts.get(HELDOUT_PROMPT, 0)),
    )
    add(
        "train_validation_rollout_overlap",
        summary["overlap_total"] == 0 and int(audit.get("split_overlap_total", -1)) == 0,
        expected=0,
        actual={"manifest": summary["overlap_total"], "audit": audit.get("split_overlap_total")},
    )

    contract_arm = arm or data_arm
    trainable_modules = list(trainable_modules or [])
    add("arm_name", bool(contract_arm), expected="non-empty", actual=contract_arm)
    if contract_arm in I_ARM_MARKERS:
        expected_marker = I_ARM_MARKERS[contract_arm]
        expected_parent = I_ARM_PARENTS[contract_arm]
        expected_source = I_ARM_MARKER_SOURCES[contract_arm]
        add("i_data_arm", data_arm == "D0", expected="D0", actual=data_arm)
        add("i_marker_mode", marker_mode == expected_marker, expected=expected_marker, actual=marker_mode)
        add("i_parent_arm", parent_arm == expected_parent, expected=expected_parent, actual=parent_arm)
        add("i_marker_source", marker_source == expected_source, expected=expected_source, actual=marker_source)
        add(
            "i_diagnostic_only",
            diagnostic_only is I_ARM_DIAGNOSTIC[contract_arm],
            expected=I_ARM_DIAGNOSTIC[contract_arm],
            actual=diagnostic_only,
        )
        add(
            "i_init_checkpoint",
            bool(init_checkpoint),
            expected="absolute checkpoint path",
            actual=init_checkpoint,
        )
        add(
            "i_init_checkpoint_sha256",
            bool(init_checkpoint_sha256 and SHA256_PATTERN.fullmatch(init_checkpoint_sha256)),
            expected="64-character init checkpoint SHA256",
            actual=init_checkpoint_sha256,
        )
        if contract_arm == "I0":
            add(
                "i_point_checkpoint_optional",
                point_checkpoint is None and point_checkpoint_sha256 is None,
                expected="not provided",
                actual={"path": point_checkpoint, "sha256": point_checkpoint_sha256},
            )
            add(
                "i_trainable_modules",
                "action_model" in trainable_modules and "point_evidence_projector" not in trainable_modules,
                expected=["action_model"],
                actual=trainable_modules,
            )
        else:
            add(
                "i_point_checkpoint",
                bool(point_checkpoint),
                expected="absolute point-head checkpoint path",
                actual=point_checkpoint,
            )
            add(
                "i_point_checkpoint_sha256",
                bool(point_checkpoint_sha256 and SHA256_PATTERN.fullmatch(point_checkpoint_sha256)),
                expected="64-character point-head checkpoint SHA256",
                actual=point_checkpoint_sha256,
            )
            if contract_arm == "I2":
                add(
                    "i_trainable_modules",
                    "action_model" in trainable_modules
                    and "point_evidence_projector" in trainable_modules,
                    expected=["action_model", "point_evidence_projector"],
                    actual=trainable_modules,
                )
            else:
                add(
                    "i_trainable_modules",
                    trainable_modules == [],
                    expected=[],
                    actual=trainable_modules,
                )
    else:
        add("d_marker_none", marker_mode == "none", expected="none", actual=marker_mode)
        add("d_counterfactual_none", counterfactual_mode == "none", expected="none", actual=counterfactual_mode)

    overlap_total = summary["overlap_total"] + int(audit.get("split_overlap_total", 0))
    contract_status = "PASS" if not failures else "INVALID"
    return {
        "schema": SCHEMA,
        "created_at": utc_now(),
        "contract_status": contract_status,
        "arm_status": "PENDING_GATE" if contract_status == "PASS" else "INVALID",
        "arm": contract_arm,
        "parent_arm": parent_arm,
        "code_commit": code_commit,
        "config_sha256": config_sha256,
        "marker_source": marker_source,
        "diagnostic_only": bool(diagnostic_only),
        "init_checkpoint": init_checkpoint,
        "init_checkpoint_sha256": init_checkpoint_sha256,
        "point_checkpoint": point_checkpoint,
        "point_checkpoint_sha256": point_checkpoint_sha256,
        "trainable_modules": trainable_modules,
        "data": {
            "data_arm": data_arm,
            "train_language_counts": train_language_counts,
            "heldout_prompt_train_count": int(train_language_counts.get(HELDOUT_PROMPT, 0)),
            "train_validation_rollout_overlap": overlap_total,
            "train_episode_count": train_episode_count,
            "carrier_episode_count": int(train_language_counts.get(CARRIER_PROMPT, 0)),
            "white_bowl_right_episode_count": int(train_language_counts.get(WHITE_BOWL_PROMPT, 0)),
            "source_language_counts": source_counts,
            "split_language_counts": audit_split_counts,
            "manifest_split_counts": summary["split_counts"],
            "sampling_weights": normalized_observed["sampling_weights"] if normalized_observed else {},
            "dataset_lengths": normalized_observed["dataset_lengths"] if normalized_observed else {},
            "expected_raw_transitions": expected_raw_transitions,
            "expected_trajectories": expected_trajectories,
        },
        "runtime": {
            "per_device_batch": per_device_batch,
            "num_processes": num_processes,
            "grad_accum": grad_accum,
            "effective_batch": effective_batch,
            "seed": seed,
        },
        "marker_mode": marker_mode,
        "counterfactual_mode": counterfactual_mode,
        "checks": checks,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--data-arms-config", type=Path, default=Path("configs/vla/goal13_data_arms.json"))
    parser.add_argument("--data-arm", choices=("D0", "D1", "D2"), required=True)
    parser.add_argument("--arm", choices=("I0", "I1", "I2", "I3"), default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-arm", default=None)
    parser.add_argument("--marker-mode", choices=("none", "oracle", "predicted", "shuffle"), default="none")
    parser.add_argument("--marker-source", choices=("none", "gt_point_manifest", "point_head"), default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--init-checkpoint-sha256", default=None)
    parser.add_argument("--point-checkpoint", default=None)
    parser.add_argument("--point-checkpoint-sha256", default=None)
    parser.add_argument("--trainable-modules", default=None, help="comma-separated module names; empty string means eval-only")
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--counterfactual-mode", choices=("none", "rank", "suff_nec"), default="none")
    parser.add_argument("--code-commit", default=None)
    parser.add_argument("--config-sha256", default=None)
    parser.add_argument("--config", type=Path, default=None, help="optional config file; SHA256 is computed if --config-sha256 is omitted")
    parser.add_argument("--observed-dataset-contract", type=Path, default=None)
    parser.add_argument("--per-device-batch", type=int, required=True)
    parser.add_argument("--grad-accum", type=int, required=True)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--expected-seed", type=int, default=17)
    parser.add_argument("--expected-effective-batch", type=int, default=4)
    parser.add_argument("--expected-carrier-count", type=int, default=48)
    parser.add_argument("--expected-white-bowl-count", type=int, default=44)
    parser.add_argument("--expected-c1-dataset-length", type=int, default=11630)
    parser.add_argument("--c1-dataset-name", default="libero_90_gate_lerobot")
    parser.add_argument("--expected-carrier-dataset-length", type=int, default=None)
    parser.add_argument("--expected-c1-sampling-weight", type=float, default=None)
    parser.add_argument("--expected-carrier-sampling-weight", type=float, default=None)
    parser.add_argument("--carrier-manifest", type=Path, default=None)
    parser.add_argument("--expected-raw-transitions", type=int, default=None)
    parser.add_argument("--expected-trajectories", type=int, default=None)
    args = parser.parse_args()

    data_arms = load_data_arms(args.data_arms_config)
    if args.data_arm not in data_arms["arms"]:
        print(f"ERROR: unknown data arm {args.data_arm}", file=sys.stderr)
        return 2
    config_sha = args.config_sha256
    if config_sha is None and args.config is not None:
        config_sha = sha256_file(args.config)
    observed = load_json(args.observed_dataset_contract) if args.observed_dataset_contract else None
    carrier_rows = load_jsonl(args.carrier_manifest) if args.carrier_manifest else None
    audit = load_json(args.audit)
    split_manifest = load_json(args.split_manifest)

    contract = evaluate_contract(
        audit=audit,
        split_manifest=split_manifest,
        data_arm=args.data_arm,
        arm_definition=data_arms["arms"][args.data_arm],
        runtime={
            "per_device_batch": args.per_device_batch,
            "num_processes": args.num_processes,
            "grad_accum": args.grad_accum,
            "seed": args.seed,
        },
        code_commit=args.code_commit,
        config_sha256=config_sha,
        observed_contract=observed,
        parent_arm=args.parent_arm,
        marker_mode=args.marker_mode,
        counterfactual_mode=args.counterfactual_mode,
        expected_source_counts=data_arms["expected_source_counts"],
        expected_train_counts=data_arms["expected_train_counts"],
        expected_manifest_split_counts=data_arms["expected_manifest_split_counts"],
        expected_carrier_count=args.expected_carrier_count,
        expected_white_bowl_count=args.expected_white_bowl_count,
        expected_c1_dataset_length=args.expected_c1_dataset_length,
        c1_dataset_name=args.c1_dataset_name,
        expected_carrier_dataset_length=args.expected_carrier_dataset_length,
        expected_c1_sampling_weight=args.expected_c1_sampling_weight,
        expected_carrier_sampling_weight=args.expected_carrier_sampling_weight,
        carrier_rows=carrier_rows,
        expected_raw_transitions=args.expected_raw_transitions,
        expected_trajectories=args.expected_trajectories,
        prompt_text=data_arms["prompts"],
        expected_seed=args.expected_seed,
        expected_effective_batch=args.expected_effective_batch,
        arm=args.arm,
        marker_source=args.marker_source,
        init_checkpoint=args.init_checkpoint,
        init_checkpoint_sha256=args.init_checkpoint_sha256,
        point_checkpoint=args.point_checkpoint,
        point_checkpoint_sha256=args.point_checkpoint_sha256,
        trainable_modules=(
            [item.strip() for item in args.trainable_modules.split(",") if item.strip()]
            if args.trainable_modules is not None
            else None
        ),
        diagnostic_only=args.diagnostic_only,
    )
    atomic_json_write(args.output, contract)
    print(
        json.dumps(
            {
                "contract_status": contract["contract_status"],
                "arm_status": contract["arm_status"],
                "failures": contract["failures"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if contract["contract_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
