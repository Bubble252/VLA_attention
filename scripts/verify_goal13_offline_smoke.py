#!/usr/bin/env python3
"""Validate a Goal13 offline eval smoke without turning it into a method gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "goal13-offline-smoke-contract-v1"
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


def requested_metrics(pair: dict[str, Any]) -> dict[str, Any]:
    return {name: pair.get(name) for name in REQUIRED_PAIR_METRICS}


def evaluate_offline_smoke(
    *,
    postrun_path: Path,
    offline_path: Path,
    expected_limit: int,
    expected_arm: str = "D0",
) -> dict[str, Any]:
    postrun = load_json(postrun_path)
    offline = load_json(offline_path)
    checks: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    def check(name: str, passed: bool, expected: Any, actual: Any, detail: str = "") -> None:
        add_check(checks, name, passed, expected=expected, actual=actual, detail=detail)
        if not passed:
            failures.append(name)

    check("postrun_status", postrun.get("postrun_status") == "PASS", "PASS", postrun.get("postrun_status"))
    check("offline_schema", offline.get("schema") == EXPECTED_OFFLINE_SCHEMA, EXPECTED_OFFLINE_SCHEMA, offline.get("schema"))
    check("offline_arm", offline.get("arm") == expected_arm, expected_arm, offline.get("arm"))

    checkpoint = offline.get("checkpoint", {})
    config = offline.get("config", {})
    check(
        "checkpoint_path",
        Path(str(checkpoint.get("path", ""))).resolve() == Path(str(postrun.get("checkpoint", ""))).resolve(),
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
        Path(str(config.get("path", ""))).resolve() == Path(str(postrun.get("config_snapshot", ""))).resolve(),
        postrun.get("config_snapshot"),
        config.get("path"),
    )
    check(
        "config_sha256",
        config.get("sha256") == postrun.get("config_snapshot_sha256"),
        postrun.get("config_snapshot_sha256"),
        config.get("sha256"),
    )
    check("num_groups", int(offline.get("num_groups", -1)) == expected_limit, expected_limit, offline.get("num_groups"))

    pair_results = offline.get("pair_results", {})
    check("destination_pairs_present", "destination" in pair_results, True, sorted(pair_results))
    check("object_pairs_present", "object" in pair_results, True, sorted(pair_results))

    metrics: dict[str, Any] = {}
    for kind in ("destination", "object"):
        pair = pair_results.get(kind, {})
        selected = requested_metrics(pair) if isinstance(pair, dict) else {}
        metrics[kind] = selected
        for name in REQUIRED_PAIR_METRICS:
            value = pair.get(name) if isinstance(pair, dict) else None
            check(
                f"{kind}_{name}_present",
                isinstance(value, (int, float)) and not isinstance(value, bool),
                "number",
                value,
            )
        check(
            f"{kind}_pair_count_positive",
            int(pair.get("pair_count", 0)) > 0 if isinstance(pair, dict) else False,
            ">0",
            pair.get("pair_count") if isinstance(pair, dict) else None,
        )

    bad_numbers = nonfinite_paths(
        {
            "pair_results": pair_results,
            "seed": offline.get("seed"),
            "num_groups": offline.get("num_groups"),
        }
    )
    check("all_metrics_finite", not bad_numbers, [], bad_numbers)
    check("region_results_absent", offline.get("region_results") is None, None, offline.get("region_results"))

    status = "SMOKE_PASS" if not failures else "INVALID"
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "postrun_status": status,
        "gate_status": "NOT_EVALUATED",
        "arm": offline.get("arm"),
        "postrun": {"path": str(postrun_path), "sha256": sha256_file(postrun_path)},
        "offline": {"path": str(offline_path), "sha256": sha256_file(offline_path)},
        "expected_limit_groups": expected_limit,
        "num_groups": offline.get("num_groups"),
        "num_pairs": {kind: pair_results.get(kind, {}).get("pair_count") for kind in ("destination", "object")},
        "metrics": metrics,
        "checks": checks,
        "failures": failures,
        "interpretation": (
            "Evaluator/checkpoint/config path is valid. Two groups and an intentionally short "
            "smoke checkpoint are not evidence for action quality or method efficacy."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--postrun", type=Path, required=True)
    parser.add_argument("--offline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-limit-groups", type=int, default=2)
    parser.add_argument("--expected-arm", default="D0")
    args = parser.parse_args()
    report = evaluate_offline_smoke(
        postrun_path=args.postrun,
        offline_path=args.offline,
        expected_limit=args.expected_limit_groups,
        expected_arm=args.expected_arm,
    )
    atomic_json_write(args.output, report)
    print(
        json.dumps(
            {
                "postrun_status": report["postrun_status"],
                "gate_status": report["gate_status"],
                "failures": report["failures"],
                "num_groups": report["num_groups"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if report["postrun_status"] == "SMOKE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
