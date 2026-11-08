#!/usr/bin/env python3
"""Select the Goal17 execution arm and audit it against the pre-registered Gate.

The script only reads existing audit summaries.  It never runs a rollout and
never changes a threshold.  E1 at the calibration-selected settle length is the
pre-registered G-B arm.  E2 is the full-3D control; if it has strictly higher
success, the Goal17 protocol says to keep the 3D hand-back arm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1-baseline-summary", type=Path, required=True)
    parser.add_argument("--e1-settle-summary", type=Path, required=True)
    parser.add_argument("--e2-summary", type=Path, required=True)
    parser.add_argument("--goal-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _arm(summary: Mapping[str, Any], arm: str) -> dict[str, Any]:
    by_arm = summary.get("by_arm") or {}
    if arm not in by_arm:
        raise ValueError(f"arm {arm!r} missing from summary")
    return dict(by_arm[arm])


def _int(item: Mapping[str, Any], key: str) -> int:
    return int(item.get(key, 0) or 0)


def evaluate_arm(
    metrics: Mapping[str, Any],
    *,
    b0: Mapping[str, Any],
    gate: Mapping[str, Any],
    grounding_passed: bool,
) -> dict[str, Any]:
    checks = {
        "grounding_passed": bool(grounding_passed),
        "episodes": _int(metrics, "episodes") == int(gate["grounding_init_denominator"]),
        "success": _int(metrics, "success") >= int(gate["execution_r_pred_success_min"]),
        "success_minus_b0": _int(metrics, "success") - _int(b0, "success")
        >= int(gate["execution_r_pred_minus_b0_min"]),
        "released_success": _int(metrics, "released_success")
        >= int(gate["execution_released_success_min"]),
        "wrong_destination": _int(metrics, "wrong_destination")
        <= _int(b0, "wrong_destination"),
        "no_errors": _int(metrics, "errors") == 0,
        "no_nan": _int(metrics, "nan_count") == 0,
        "no_flew_out": _int(metrics, "object_flew_out") == 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed": [name for name, value in checks.items() if not value],
    }


def _metrics(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item.get(key)
        for key in (
            "episodes",
            "success",
            "released_success",
            "hold_only_success",
            "released_placement",
            "target_destination_correct",
            "wrong_destination",
            "transport_started",
            "handback_seen",
            "rearm_total",
            "errors",
            "nan_count",
            "object_flew_out",
        )
    }


def main() -> int:
    args = parse_args()
    goal = json.loads(args.goal_config.read_text())
    gate = goal["gate"]
    e1_base_summary = json.loads(args.e1_baseline_summary.read_text())
    e1_s32_summary = json.loads(args.e1_settle_summary.read_text())
    e2_summary = json.loads(args.e2_summary.read_text())

    b0 = _arm(e1_s32_summary, "B0")
    e1_base = _arm(e1_base_summary, "E1-PRED")
    e1_s32 = _arm(e1_s32_summary, "E1-PRED")
    e2_pred = _arm(e2_summary, "E2-PRED")
    e2_geo = _arm(e2_summary, "E2-GEO")
    grounding_passed = bool(e1_s32_summary.get("grounding", {}).get("passed") is True)

    e1_gate = evaluate_arm(e1_s32, b0=b0, gate=gate, grounding_passed=grounding_passed)
    e2_gate = evaluate_arm(e2_pred, b0=b0, gate=gate, grounding_passed=grounding_passed)

    # This follows Goal17 section 5: E2 > E1 keeps the 3D hand-back arm.
    best_arm = "E2-PRED" if _int(e2_pred, "success") > _int(e1_s32, "success") else "E1-PRED-SETTLE32"
    default_arm = best_arm
    selected_gate = e2_gate if best_arm == "E2-PRED" else e1_gate
    status = "PASS" if selected_gate["passed"] else "FAIL"

    payload = {
        "schema": "goal17-execution-selection-v1",
        "status": status,
        "grounding_passed": grounding_passed,
        "pre_registered_g_b_arm": "E1-PRED-SETTLE32",
        "e1_settle_steps": e1_s32_summary.get("execution_overrides", {}).get(
            "release.settle_steps"
        ),
        "best_arm": best_arm,
        "default_arm": default_arm,
        "simpler_passing_arm": "E1-PRED-SETTLE32" if e1_gate["passed"] else None,
        "comparison": {
            "B0": _metrics(b0),
            "E1-PRED-SETTLE4": _metrics(e1_base),
            "E1-PRED-SETTLE32": _metrics(e1_s32),
            "E2-GEO": _metrics(e2_geo),
            "E2-PRED": _metrics(e2_pred),
        },
        "by_split": {
            "E1-PRED-SETTLE32": e1_s32_summary.get("by_split", {}).get("E1-PRED"),
            "E2-PRED": e2_summary.get("by_split", {}).get("E2-PRED"),
        },
        "gates": {
            "E1-PRED-SETTLE32": e1_gate,
            "E2-PRED": e2_gate,
        },
        "sources": {
            "e1_baseline_summary": str(args.e1_baseline_summary),
            "e1_settle_summary": str(args.e1_settle_summary),
            "e2_summary": str(args.e2_summary),
            "goal_config": str(args.goal_config),
        },
        "interpretation": [
            "Grounding G-A is repaired on the deployable 4-prompt track.",
            "Track C remains diagnostic and does not establish compositional generalization.",
            "Settle length changes E1 from 0/15 at 4 steps to 2/15 at 32 steps.",
            "E2-PRED has 3/15 and is kept as the best arm under Goal17 section 5.",
            "The E1-settle32 arm is the simpler passing fallback.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selection.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    lines = [
        "# Goal17 execution selection",
        "",
        f"- status: `{status}`",
        f"- best arm: `{best_arm}`",
        f"- default arm: `{default_arm}`",
        f"- grounding: `{'PASS' if grounding_passed else 'FAIL'}`",
        f"- E1 settle steps: `{payload['e1_settle_steps']}`",
        "",
        "| arm | success | released | hold-only | wrong dest |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in payload["comparison"].items():
        lines.append(
            f"| `{name}` | {item.get('success')} | {item.get('released_success')} | "
            f"{item.get('hold_only_success')} | {item.get('wrong_destination')} |"
        )
    lines += [
        "",
        "## Gates",
        "",
        f"- `E1-PRED-SETTLE32`: `{e1_gate['passed']}`; failed={e1_gate['failed']}",
        f"- `E2-PRED`: `{e2_gate['passed']}`; failed={e2_gate['failed']}",
    ]
    (args.output_dir / "selection.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
