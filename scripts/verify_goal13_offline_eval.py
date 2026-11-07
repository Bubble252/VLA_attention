#!/usr/bin/env python3
"""Validate a full Goal13 offline evaluation against its postrun contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "goal13-offline-eval-contract-v1"
EXPECTED_OFFLINE_SCHEMA = "goal12-c1-counterfactual-offline-v1"
REQUIRED_PAIR_METRICS = (
    "pair_count",
    "correct_mae",
    "margin_mean",
    "action_amplitude_ratio",
    "translation_amplitude_ratio",
    "rotation_amplitude_ratio",
    "gripper_amplitude_ratio",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def nonfinite_paths(value: Any, prefix: tuple[str, ...] = ()) -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            failures.extend(nonfinite_paths(item, prefix + (str(key),)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            failures.extend(nonfinite_paths(item, prefix + (str(index),)))
    elif isinstance(value, float) and not math.isfinite(value):
        failures.append(".".join(prefix))
    return failures


def evaluate_offline_eval(
    *,
    postrun_path: Path,
    offline_path: Path,
    expected_groups: int,
    expected_destination_pairs: int,
    expected_object_pairs: int,
    expected_arm: str = "D0",
    expected_postrun_arm: str | None = None,
) -> dict[str, Any]:
    postrun = load_json(postrun_path)
    offline = load_json(offline_path)
    checks: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    def check(
        name: str,
        passed: bool,
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
        if not passed:
            failures.append(name)

    check(
        "postrun_status",
        postrun.get("postrun_status") == "PASS",
        "PASS",
        postrun.get("postrun_status"),
    )
    check(
        "postrun_arm",
        postrun.get("arm") == (expected_postrun_arm or expected_arm),
        expected_postrun_arm or expected_arm,
        postrun.get("arm"),
    )
    check(
        "offline_schema",
        offline.get("schema") == EXPECTED_OFFLINE_SCHEMA,
        EXPECTED_OFFLINE_SCHEMA,
        offline.get("schema"),
    )
    check("offline_arm", offline.get("arm") == expected_arm, expected_arm, offline.get("arm"))

    checkpoint = offline.get("checkpoint", {})
    config = offline.get("config", {})
    check(
        "checkpoint_path",
        Path(str(checkpoint.get("path", ""))).resolve()
        == Path(str(postrun.get("checkpoint", ""))).resolve(),
        postrun.get("checkpoint"),
        checkpoint.get("path"),
    )
    check(
        "checkpoint_sha256",
        checkpoint.get("sha256") == postrun.get("checkpoint_sha256"),
        postrun.get("checkpoint_sha256"),
        checkpoint.get("sha256"),
    )
    check(
        "config_path",
        Path(str(config.get("path", ""))).resolve()
        == Path(str(postrun.get("config_snapshot", ""))).resolve(),
        postrun.get("config_snapshot"),
        config.get("path"),
    )
    check(
        "config_sha256",
        config.get("sha256") == postrun.get("config_snapshot_sha256"),
        postrun.get("config_snapshot_sha256"),
        config.get("sha256"),
    )
    check("num_groups", offline.get("num_groups") == expected_groups, expected_groups, offline.get("num_groups"))

    pair_results = offline.get("pair_results", {})
    expected_pairs = {
        "destination": expected_destination_pairs,
        "object": expected_object_pairs,
    }
    check("pair_result_kinds", sorted(pair_results) == sorted(expected_pairs), sorted(expected_pairs), sorted(pair_results))

    metrics: dict[str, Any] = {}
    for kind, expected_count in expected_pairs.items():
        pair = pair_results.get(kind, {})
        metrics[kind] = (
            {name: pair.get(name) for name in REQUIRED_PAIR_METRICS}
            if isinstance(pair, dict)
            else {}
        )
        check(
            f"{kind}_pair_count",
            int(pair.get("pair_count", -1)) == expected_count if isinstance(pair, dict) else False,
            expected_count,
            pair.get("pair_count") if isinstance(pair, dict) else None,
        )
        for name in REQUIRED_PAIR_METRICS:
            value = pair.get(name) if isinstance(pair, dict) else None
            check(
                f"{kind}_{name}_present",
                isinstance(value, (int, float)) and not isinstance(value, bool),
                "number",
                value,
            )

    bad_numbers = nonfinite_paths(
        {
            "pair_results": pair_results,
            "seed": offline.get("seed"),
            "num_groups": offline.get("num_groups"),
        }
    )
    check("all_metrics_finite", not bad_numbers, [], bad_numbers)

    status = "PASS" if not failures else "INVALID"
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract_status": status,
        "gate_status": "DESCRIPTIVE" if status == "PASS" else "NOT_EVALUATED",
        "arm": offline.get("arm"),
        "postrun": {"path": str(postrun_path), "sha256": sha256_file(postrun_path)},
        "offline": {"path": str(offline_path), "sha256": sha256_file(offline_path)},
        "expected": {
            "groups": expected_groups,
            "destination_pairs": expected_destination_pairs,
            "object_pairs": expected_object_pairs,
        },
        "num_groups": offline.get("num_groups"),
        "num_pairs": {
            kind: pair_results.get(kind, {}).get("pair_count")
            for kind in expected_pairs
        },
        "metrics": metrics,
        "offline_gate": offline.get("gate"),
        "checks": checks,
        "failures": failures,
        "interpretation": (
            "The full paired evaluator ran on the expected group and pair counts. "
            "Offline gate booleans are descriptive; closed-loop destination evidence "
            "is still required before any method claim."
        ),
        "expected_postrun_arm": expected_postrun_arm or expected_arm,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--postrun", type=Path, required=True)
    parser.add_argument("--offline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-groups", type=int, required=True)
    parser.add_argument("--expected-destination-pairs", type=int, required=True)
    parser.add_argument("--expected-object-pairs", type=int, required=True)
    parser.add_argument("--expected-arm", default="D0")
    parser.add_argument("--expected-postrun-arm", default=None)
    args = parser.parse_args()
    report = evaluate_offline_eval(
        postrun_path=args.postrun,
        offline_path=args.offline,
        expected_groups=args.expected_groups,
        expected_destination_pairs=args.expected_destination_pairs,
        expected_object_pairs=args.expected_object_pairs,
        expected_arm=args.expected_arm,
        expected_postrun_arm=args.expected_postrun_arm,
    )
    atomic_json_write(args.output, report)
    print(
        json.dumps(
            {
                "contract_status": report["contract_status"],
                "gate_status": report["gate_status"],
                "failures": report["failures"],
                "num_groups": report["num_groups"],
                "num_pairs": report["num_pairs"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if report["contract_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
