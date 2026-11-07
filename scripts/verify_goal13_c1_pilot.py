#!/usr/bin/env python3
"""Validate the frozen 15-init C1 pilot against a Goal13 postrun contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "goal13-c1-pilot-contract-v1"


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


def evaluate_c1_pilot(
    *,
    postrun_path: Path,
    summary_path: Path,
    expected_groups: int = 15,
    expected_arm: str = "D0",
) -> dict[str, Any]:
    postrun = load_json(postrun_path)
    summary = load_json(summary_path)
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
    check("postrun_arm", postrun.get("arm") == expected_arm, expected_arm, postrun.get("arm"))
    check(
        "num_init_groups",
        summary.get("num_init_groups") == expected_groups,
        expected_groups,
        summary.get("num_init_groups"),
    )
    check(
        "num_episodes",
        summary.get("num_episodes") == expected_groups * 4,
        expected_groups * 4,
        summary.get("num_episodes"),
    )

    seen = summary.get("seen", {})
    white_right = summary.get("white_right", {})
    check("seen_episodes", seen.get("episodes") == expected_groups * 3, expected_groups * 3, seen.get("episodes"))
    check("seen_group_any_range", 0 <= int(seen.get("group_any_successes", -1)) <= expected_groups, [0, expected_groups], seen.get("group_any_successes"))
    check("seen_group_all_range", 0 <= int(seen.get("group_all_successes", -1)) <= expected_groups, [0, expected_groups], seen.get("group_all_successes"))
    check("white_right_episodes", white_right.get("episodes") == expected_groups, expected_groups, white_right.get("episodes"))
    check("white_right_success_range", 0 <= int(white_right.get("successes", -1)) <= expected_groups, [0, expected_groups], white_right.get("successes"))
    check(
        "white_right_strict_destination_range",
        0 <= int(white_right.get("strict_destination_successes", -1)) <= expected_groups,
        [0, expected_groups],
        white_right.get("strict_destination_successes"),
    )

    groups = summary.get("groups", [])
    check("group_rows_count", isinstance(groups, list) and len(groups) == expected_groups, expected_groups, len(groups) if isinstance(groups, list) else None)
    group_ids = [str(row.get("shared_init_id")) for row in groups] if isinstance(groups, list) else []
    check("group_ids_unique", len(group_ids) == len(set(group_ids)), "unique", len(group_ids) - len(set(group_ids)))

    bad_numbers = nonfinite_paths(
        {
            "seen": seen,
            "white_right": white_right,
            "groups": groups,
        }
    )
    check("all_metrics_finite", not bad_numbers, [], bad_numbers)

    status = "PASS" if not failures else "INVALID"
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract_status": status,
        "gate_status": "NOT_EVALUATED",
        "arm": postrun.get("arm"),
        "postrun": {"path": str(postrun_path), "sha256": sha256_file(postrun_path)},
        "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        "expected": {"init_groups": expected_groups, "episodes": expected_groups * 4},
        "seen": seen,
        "white_right": white_right,
        "checks": checks,
        "failures": failures,
        "interpretation": (
            "This fifteen-init pilot is a screening run. It is not a formal method "
            "claim; confirmation requires 30/50 inits and multiple seeds."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--postrun", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-groups", type=int, default=15)
    parser.add_argument("--expected-arm", default="D0")
    args = parser.parse_args()
    report = evaluate_c1_pilot(
        postrun_path=args.postrun,
        summary_path=args.summary,
        expected_groups=args.expected_groups,
        expected_arm=args.expected_arm,
    )
    atomic_json_write(args.output, report)
    print(
        json.dumps(
            {
                "contract_status": report["contract_status"],
                "failures": report["failures"],
                "seen": report["seen"],
                "white_right": report["white_right"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if report["contract_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
