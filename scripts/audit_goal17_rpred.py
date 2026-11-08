#!/usr/bin/env python3
"""Goal17 E1/E2 execution audit with predicted destination points.

E1 is the simplified R-NAIVE transport path (xy only, explicit release); E2 is
the full 3D R-GEO path.  Both can consume either an oracle destination (control)
or the Goal17 point-head sidecar.  R-PRED never receives an oracle fallback.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_goal16_reguide_rebinding import (  # noqa: E402
    CALIBRATION_INIT_IDS,
    EXPECTED_INIT_COUNT,
    _by_split,
    _grounding_rows,
    _summary,
    label_split,
    run_episode,
    select_rows,
    sha256_file,
    validate_config,
)
from scripts.eval_c1_counterfactual import read_manifest, validate_manifest  # noqa: E402
from scripts.eval_e1_failure_atlas import goal_info_from_problem  # noqa: E402
from scripts.eval_j0_rollout import load_policy  # noqa: E402
from scripts.eval_oft_single_checkpoint import (  # noqa: E402
    checkpoint_fingerprint,
    resolve_checkpoint,
)

EXPECTED_EXECUTION_CONFIG_SHA256 = (
    "34731e83e56d6c2ca075bdaecbbbb87671379053194b28a4561c06271ea8e502"
)
EXPECTED_MANIFEST_SHA256 = (
    "e9564ca9aef800e618e3fdb5d144892eabe9471815bbd24bae3140583946f45e"
)
EXPECTED_CHECKPOINT_FINGERPRINT = (
    "747a28690383ca45e795518121e9aed3f4610265c14618a69f203922b4238e20"
)
EXPECTED_STATISTICS_SHA256 = (
    "6d5ebb47bc5132df581aae0481d529d04a0821c331859c05a9818d78156c9538"
)
ARM_ORDER = ("B0", "E1-ORACLE", "E1-PRED", "E2-GEO", "E2-PRED")
CONTROL_ARM = {
    "B0": "B0",
    "E1-ORACLE": "R-NAIVE",
    "E1-PRED": "R-NAIVE",
    "E2-GEO": "R-GEO",
    "E2-PRED": "R-GEO",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--goal-config", type=Path, required=True)
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--grounding-sidecar", type=Path, required=True)
    parser.add_argument("--split-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", default=",".join(ARM_ORDER))
    parser.add_argument("--max-inits", type=int, default=0)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=None,
        help="Override release.settle_steps for the E1 release/settle diagnostic.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-open-loop-steps", type=int, default=8)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--center-crop", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _int(item: Mapping[str, Any], key: str) -> int:
    return int(item.get(key, 0) or 0)


def evaluate_goal17_gate(
    by_arm: Mapping[str, Mapping[str, Any]],
    *,
    arms: Sequence[str],
    init_count: int,
    goal_config: Mapping[str, Any],
    grounding_status: Mapping[str, Any],
) -> dict[str, Any]:
    gate = goal_config["gate"]
    checks: dict[str, bool] = {}
    for arm in arms:
        item = by_arm.get(arm, {})
        checks[f"{arm}_rows"] = _int(item, "episodes") == int(init_count)
        checks[f"{arm}_clean"] = bool(
            _int(item, "errors") == 0
            and _int(item, "nan_count") == 0
            and _int(item, "object_flew_out") == 0
        )
    checks["grounding_passed"] = bool(grounding_status.get("passed") is True)
    if "B0" in by_arm and "E1-PRED" in by_arm:
        checks["e1_r_pred_success"] = (
            bool(grounding_status.get("passed") is True)
            and _int(by_arm["E1-PRED"], "success")
            >= int(gate["execution_r_pred_success_min"])
        )
        checks["e1_r_pred_minus_b0"] = (
            bool(grounding_status.get("passed") is True)
            and _int(by_arm["E1-PRED"], "success") - _int(by_arm["B0"], "success")
            >= int(gate["execution_r_pred_minus_b0_min"])
        )
        checks["e1_r_pred_released"] = (
            bool(grounding_status.get("passed") is True)
            and _int(by_arm["E1-PRED"], "released_success")
            >= int(gate["execution_released_success_min"])
        )
        checks["e1_r_pred_wrong_destination"] = (
            _int(by_arm["E1-PRED"], "wrong_destination")
            <= _int(by_arm["B0"], "wrong_destination")
        )
    else:
        checks["e1_r_pred_success"] = False
        checks["e1_r_pred_minus_b0"] = False
        checks["e1_r_pred_released"] = False
        checks["e1_r_pred_wrong_destination"] = False
    checks["no_errors"] = all(_int(by_arm.get(arm, {}), "errors") == 0 for arm in arms)
    checks["no_nan"] = all(_int(by_arm.get(arm, {}), "nan_count") == 0 for arm in arms)
    checks["no_flew_out"] = all(
        _int(by_arm.get(arm, {}), "object_flew_out") == 0 for arm in arms
    )
    passed = all(checks.values())
    if not checks["grounding_passed"]:
        status = "INVALID"
    elif passed:
        status = "PASS"
    else:
        status = "FAIL"
    return {
        "status": status,
        "passed": passed,
        "checks": checks,
        "failed": [name for name, value in checks.items() if not value],
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Goal17 E1/E2 execution audit",
        "",
        f"- status: `{summary['status']}`",
        f"- grounding: `{summary['grounding']['status']}`",
        f"- manifest SHA256: `{summary['manifest_sha256']}`",
        f"- sidecar SHA256: `{summary['grounding']['sidecar_sha256']}`",
        "",
        "| arm | episodes | success | released | hold-only | wrong dest | errors |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in summary["arms_order"]:
        item = summary["by_arm"][arm]
        lines.append(
            f"| `{arm}` | {item.get('episodes')} | {item.get('success')} | "
            f"{item.get('released_success')} | {item.get('hold_only_success')} | "
            f"{item.get('wrong_destination')} | {item.get('errors')} |"
        )
    lines += ["", "Failed checks:", ""]
    lines += [f"- `{name}`" for name in summary["gate"]["failed"]] or ["- none"]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    goal_config = json.loads(args.goal_config.read_text())
    if goal_config.get("status") != "frozen_before_measurement":
        raise ValueError("Goal17 config is not frozen")
    if goal_config.get("oracle_fallback_allowed") is not False:
        raise ValueError("oracle fallback must be disabled")
    execution_sha = sha256_file(args.execution_config)
    if execution_sha != EXPECTED_EXECUTION_CONFIG_SHA256:
        raise ValueError(f"execution config SHA mismatch: {execution_sha}")
    execution_config = validate_config(json.loads(args.execution_config.read_text()))
    execution_overrides: dict[str, Any] = {}
    if args.settle_steps is not None:
        if args.settle_steps < 1:
            raise ValueError("--settle-steps must be positive")
        execution_config["release"]["settle_steps"] = int(args.settle_steps)
        execution_overrides["release.settle_steps"] = int(args.settle_steps)
    manifest_sha = sha256_file(args.manifest)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"manifest SHA mismatch: {manifest_sha}")

    selected_arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    if any(arm not in ARM_ORDER for arm in selected_arms):
        raise ValueError(f"unknown arm in {selected_arms}")
    if not args.allow_partial and tuple(selected_arms) not in (
        tuple(ARM_ORDER[:3]),
        tuple(ARM_ORDER[3:]),
    ):
        raise ValueError("run must be B0,E1-ORACLE,E1-PRED or E2-GEO,E2-PRED")

    import torch
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as bddl_utils
    from experiments.robot.libero.run_libero_eval import GenerateConfig

    rows = read_manifest(args.manifest)
    validate_manifest(rows)
    selected = label_split(select_rows(rows, args.max_inits), allow_partial=args.allow_partial)
    if not args.allow_partial and len(selected) != EXPECTED_INIT_COUNT:
        raise ValueError(f"expected {EXPECTED_INIT_COUNT} rows, got {len(selected)}")

    split_audit = json.loads(args.split_audit.read_text())
    init_shas = {str(row["init_state_sha256"]) for row in selected}
    split_ok = bool(
        split_audit.get("status") == "PASS"
        and split_audit.get("gates", {}).get("white_right_train_zero") is True
        and split_audit.get("gates", {}).get("p0_eval_disjoint_from_train") is True
        and split_audit.get("gates", {}).get("p0_eval_disjoint_from_validation") is True
        and set(split_audit.get("p0_white_right", {}).get("init_state_shas", []))
        == init_shas
    )
    if not args.allow_partial and not split_ok:
        raise ValueError("split audit does not certify eval isolation")

    grounding_rows = _grounding_rows(args.grounding_sidecar)
    selected_init_ids = {str(row["shared_init_id"]) for row in selected}
    missing_grounding = selected_init_ids - set(grounding_rows)
    if missing_grounding:
        raise ValueError(
            f"grounding sidecar does not cover selected inits: {sorted(missing_grounding)}"
        )
    if not args.allow_partial and set(grounding_rows) != selected_init_ids:
        raise ValueError("grounding sidecar must cover exactly the full eval set")
    grounding_summary = json.loads((args.grounding_sidecar.parent / "summary.json").read_text())
    sidecar_sha = sha256_file(args.grounding_sidecar)
    if grounding_summary.get("sidecar_sha256") != sidecar_sha:
        raise ValueError("grounding sidecar SHA mismatch")
    grounding_status = {
        "status": "PASS" if grounding_summary.get("passed") is True else "INVALID",
        "passed": grounding_summary.get("passed") is True,
        "sidecar_sha256": sidecar_sha,
        "summary": str(args.grounding_sidecar.parent / "summary.json"),
    }

    layout = resolve_checkpoint(args.checkpoint)
    fingerprint = checkpoint_fingerprint(layout["fingerprint_files"])
    if fingerprint != EXPECTED_CHECKPOINT_FINGERPRINT:
        raise ValueError(f"checkpoint fingerprint mismatch: {fingerprint}")
    statistics_sha = sha256_file(args.statistics)
    if statistics_sha != EXPECTED_STATISTICS_SHA256:
        raise ValueError(f"statistics SHA256 mismatch: {statistics_sha}")

    policy, head, proprio, processor, _layout, _statistics = load_policy(
        args.model, args.checkpoint, args.statistics
    )
    cfg = GenerateConfig(
        pretrained_checkpoint=str(args.model),
        task_suite_name="libero_90",
        unnorm_key="project_train",
        num_open_loop_steps=args.num_open_loop_steps,
        num_steps_wait=args.num_steps_wait,
        center_crop=args.center_crop,
        use_proprio=True,
        num_images_in_input=2,
        use_wandb=False,
        seed=args.seed,
        env_img_res=args.resolution,
    )
    first = selected[0]
    bddl_path = Path(str(first["bddl_path"]))
    goal_info = goal_info_from_problem(bddl_utils.robosuite_parse_problem(str(bddl_path)))
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=args.resolution,
        camera_widths=args.resolution,
    )
    env.seed(0)

    args.output.mkdir(parents=True)
    trace_root = args.output / "traces"
    trace_root.mkdir()
    all_rows: list[dict[str, Any]] = []
    by_arm_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in selected_arms}
    started = time.time()
    try:
        for arm_id in selected_arms:
            arm_trace_root = trace_root / arm_id
            arm_trace_root.mkdir()
            for row in selected:
                grounding = grounding_rows.get(str(row["shared_init_id"]))
                destination_override = None
                if arm_id in {"E1-PRED", "E2-PRED"}:
                    if grounding is None or grounding.get("destination_position_xyz") is None:
                        raise ValueError(f"{row['shared_init_id']}: missing predicted destination")
                    destination_override = grounding["destination_position_xyz"]
                result = run_episode(
                    cfg=cfg,
                    env=env,
                    row=row,
                    policy=policy,
                    head=head,
                    proprio=proprio,
                    processor=processor,
                    config=execution_config,
                    arm_id=CONTROL_ARM[arm_id],
                    control_arm_id=CONTROL_ARM[arm_id],
                    arm_label=arm_id,
                    destination_override=destination_override,
                    goal_info=goal_info,
                    image_size=args.image_size,
                    use_film=False,
                    grounding=grounding,
                )
                records = result.pop("records")
                trace_path = arm_trace_root / f"{row['row_id']}.jsonl"
                with trace_path.open("w") as handle:
                    for record in records:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                result["trace_path"] = str(trace_path)
                all_rows.append(result)
                by_arm_rows[arm_id].append(result)
                print(
                    json.dumps(
                        {
                            "arm": arm_id,
                            "row_id": row["row_id"],
                            "success": result["success"],
                            "released_success": result["released_success"],
                            "wrong_destination": result["wrong_destination"],
                            "destination_source": result["destination_source"],
                            "error": result["error"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        env.close()

    by_arm = {arm: _summary(by_arm_rows[arm]) for arm in selected_arms}
    by_split = {arm: _by_split(all_rows, arm) for arm in selected_arms}
    if args.allow_partial:
        gate_result = {
            "status": "SMOKE",
            "passed": None,
            "checks": {},
            "failed": [],
        }
    elif "E1-PRED" in selected_arms:
        gate_result = evaluate_goal17_gate(
            by_arm,
            arms=selected_arms,
            init_count=len(selected),
            goal_config=goal_config,
            grounding_status=grounding_status,
        )
    else:
        gate_result = {
            "status": "DIAGNOSTIC",
            "passed": None,
            "checks": {
                f"{arm}_rows": _int(by_arm[arm], "episodes") == len(selected)
                for arm in selected_arms
            }
            | {
                f"{arm}_clean": bool(
                    _int(by_arm[arm], "errors") == 0
                    and _int(by_arm[arm], "nan_count") == 0
                    and _int(by_arm[arm], "object_flew_out") == 0
                )
                for arm in selected_arms
            },
            "failed": [],
        }
    summary = {
        "schema": "goal17-execution-audit-v1",
        "status": gate_result["status"],
        "stage": "Goal17",
        "audit": "e1_e2_predicted_destination",
        "note": "E1 is R-NAIVE xy transport; E2 is full 3D R-GEO. Oracle is never used by E1-PRED/E2-PRED.",
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_sha,
        "execution_config": str(args.execution_config),
        "execution_config_sha256": execution_sha,
        "execution_overrides": execution_overrides,
        "goal_config": str(args.goal_config),
        "goal_config_sha256": sha256_file(args.goal_config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_fingerprint": fingerprint,
        "statistics": str(args.statistics),
        "statistics_sha256": statistics_sha,
        "split_audit": str(args.split_audit),
        "split_audit_sha256": sha256_file(args.split_audit),
        "grounding": grounding_status,
        "init_count": len(selected),
        "calibration_init_count": sum(
            str(row["shared_init_id"]) in CALIBRATION_INIT_IDS for row in selected
        ),
        "arms_order": selected_arms,
        "by_arm": by_arm,
        "by_split": by_split,
        "gate": gate_result,
        "elapsed_sec": time.time() - started,
    }
    with (args.output / "episodes.jsonl").open("w") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    (args.output / "summary.md").write_text(_markdown(summary))
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if gate_result["status"] in {"PASS", "SMOKE", "DIAGNOSTIC"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
