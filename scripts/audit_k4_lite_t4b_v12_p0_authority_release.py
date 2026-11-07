#!/usr/bin/env python3
"""K4-lite T4b v12 P0: lateral authority with release and settle.

P0 asks one question only: after combining v10's bounded lateral authority with
v8/v8.1's release/settle, does ``white_right`` produce real released successes?
This is a diagnostic upper-bound experiment.  It does not train, and privileged
object/plate positions are used only to trigger the diagnostic controller.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_k4_lite_t4b_v8_release_settle import (  # noqa: E402
    parse_args,
    run_episode,
    select_white_right,
    sha256_file,
    summarize_arm,
    validate_metric_amendment,
)
from scripts.eval_c1_counterfactual import read_manifest, validate_manifest  # noqa: E402
from scripts.eval_e1_failure_atlas import goal_info_from_problem  # noqa: E402
from scripts.eval_j0_rollout import load_policy  # noqa: E402
from scripts.eval_oft_single_checkpoint import (  # noqa: E402
    checkpoint_fingerprint,
    resolve_checkpoint,
)


CONFIG_VERSION = "k4_lite_t4b_v12_p0_authority_release_v1"
EXPECTED_CONFIG_SHA256 = "ecb501f86b4b795d94429cc17ce21bb2461014b64a54b3b006f2c9d344cef52d"
EXPECTED_MANIFEST_SHA256 = (
    "e9564ca9aef800e618e3fdb5d144892eabe9471815bbd24bae3140583946f45e"
)
EXPECTED_CHECKPOINT_FINGERPRINT = (
    "747a28690383ca45e795518121e9aed3f4610265c14618a69f203922b4238e20"
)
EXPECTED_STATISTICS_SHA256 = (
    "6d5ebb47bc5132df581aae0481d529d04a0821c331859c05a9818d78156c9538"
)
ARM_ORDER = ("B0", "A05", "A10", "A20")
EXPECTED_CAPS = {"B0": 0.0, "A05": 0.05, "A10": 0.10, "A20": 0.20}
CALIBRATION_INIT_IDS = frozenset(
    {
        "init_0e94a4e6b9622c6b",
        "init_2edf49999eb35566",
        "init_3500df19a4d3b5a1",
    }
)
EXPECTED_INIT_COUNT = 15
EXPECTED_CONFIRMATION_COUNT = 12
RELEASE_XY_M = 0.03
RELEASE_Z_M = 0.05
RELEASE_STREAK_STEPS = 3
SETTLE_STEPS = 4


def _require_number(
    cfg: Mapping[str, Any],
    key: str,
    expected: float,
    *,
    label: str,
) -> None:
    if key not in cfg:
        raise ValueError(f"missing {label}.{key}")
    try:
        actual = float(cfg[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}.{key}: {cfg[key]!r}") from exc
    if abs(actual - float(expected)) > 1e-12:
        raise ValueError(f"{label}.{key} must stay at {expected}, got {actual}")


def validate_gate(gate: dict[str, Any], init_count: int) -> dict[str, Any]:
    if gate.get("version") != CONFIG_VERSION:
        raise ValueError(f"unexpected gate version: {gate.get('version')!r}")
    if gate.get("status") != "frozen_before_measurement":
        raise ValueError(f"gate is not frozen: {gate.get('status')!r}")
    if int(gate.get("init_count", -1)) != int(init_count):
        raise ValueError("init_count mismatch")
    if int(init_count) != EXPECTED_INIT_COUNT:
        raise ValueError(f"P0 requires exactly {EXPECTED_INIT_COUNT} inits")
    if int(gate.get("confirmation_init_count", -1)) != EXPECTED_CONFIRMATION_COUNT:
        raise ValueError("confirmation_init_count must stay at 12")
    if set(gate.get("calibration_init_ids") or []) != set(CALIBRATION_INIT_IDS):
        raise ValueError("calibration_init_ids changed")
    if gate.get("target_object") != "porcelain_mug_1":
        raise ValueError("target_object must stay at porcelain_mug_1")
    if gate.get("target_receptacle") != "plate_2":
        raise ValueError("target_receptacle must stay at plate_2")
    if int(gate.get("max_settle_steps", -1)) != SETTLE_STEPS:
        raise ValueError("max_settle_steps must stay at 4")
    if gate.get("allow_post_done_settle") is not True:
        raise ValueError("allow_post_done_settle must be true")
    if int(gate.get("seed", -1)) != 7:
        raise ValueError("P0 seed must stay at 7")

    metrics = gate.get("metrics") or {}
    expected_metrics = {
        "placement_xy_m": 0.15,
        "placement_z_m": 0.05,
        "gripper_open_width": 0.02,
        "object_retained_min_z": 0.25,
        "official_success_xy_m": 0.03,
        "legacy_target_zone_3d_m": 0.15,
        "flew_out_max_displacement_m": 2.0,
        "flew_out_min_target_z_m": 0.0,
        "flew_out_max_target_z_m": 2.0,
    }
    for key, value in expected_metrics.items():
        _require_number(metrics, key, value, label="metrics")

    arms = gate.get("arms")
    if not isinstance(arms, dict) or tuple(arms) != ARM_ORDER:
        raise ValueError(f"gate arms must be exactly {ARM_ORDER}")

    b0 = arms["B0"]
    if b0.get("activation") != "after_lift":
        raise ValueError("B0.activation must stay at after_lift")
    _require_number(b0, "lift_threshold_m", 1_000_000_000.0, label="B0")
    _require_number(b0, "lateral_bias", 0.0, label="B0")
    _require_number(b0, "fine_cap", 0.0, label="B0")
    if b0.get("lateral_mode") != "additive_y":
        raise ValueError("B0.lateral_mode must stay at additive_y")
    if b0.get("release_at_target") is not False:
        raise ValueError("B0 must not release")
    if b0.get("hold_gripper") is not False:
        raise ValueError("B0 must not hold the gripper")
    if int(b0.get("settle_steps", -1)) != 0:
        raise ValueError("B0.settle_steps must stay at 0")
    if int(b0.get("release_streak_steps", -1)) != 0:
        raise ValueError("B0.release_streak_steps must stay at 0")

    for arm_id in ("A05", "A10", "A20"):
        arm = arms[arm_id]
        _require_number(arm, "fine_cap", EXPECTED_CAPS[arm_id], label=arm_id)
        if arm.get("activation") != "first_grasp":
            raise ValueError(f"{arm_id}.activation must stay at first_grasp")
        _require_number(arm, "grasp_gripper_width", 0.04, label=arm_id)
        if int(arm.get("activation_delay_steps", -1)) != 2:
            raise ValueError(f"{arm_id}.activation_delay_steps must stay at 2")
        if arm.get("lateral_mode") != "absolute_xy":
            raise ValueError(f"{arm_id}.lateral_mode must stay at absolute_xy")
        _require_number(arm, "coarse_xy_m", 0.1, label=arm_id)
        _require_number(arm, "coarse_cap", 0.2, label=arm_id)
        _require_number(arm, "gain", 2.0, label=arm_id)
        _require_number(arm, "deadband_m", 0.003, label=arm_id)
        if arm.get("stop_in_success_radius") is not True:
            raise ValueError(f"{arm_id}.stop_in_success_radius must be true")
        _require_number(arm, "stop_xy_m", 0.005, label=arm_id)
        if arm.get("release_at_target") is not True:
            raise ValueError(f"{arm_id} must release at target")
        if arm.get("hold_gripper") is not True:
            raise ValueError(f"{arm_id} must hold the gripper before release")
        _require_number(arm, "release_xy_m", RELEASE_XY_M, label=arm_id)
        _require_number(arm, "release_z_m", RELEASE_Z_M, label=arm_id)
        if int(arm.get("release_streak_steps", -1)) != RELEASE_STREAK_STEPS:
            raise ValueError(f"{arm_id}.release_streak_steps must stay at 3")
        if int(arm.get("release_defer_steps", -1)) != 0:
            raise ValueError(f"{arm_id}.release_defer_steps must stay at 0")
        if int(arm.get("settle_steps", -1)) != SETTLE_STEPS:
            raise ValueError(f"{arm_id}.settle_steps must stay at 4")

    frozen = gate.get("gate") or {}
    expected_gate = {
        "B0_success_max": 0,
        "A10_success_min": 5,
        "A10_confirmation_success_min": 2,
        "A10_released_success_min": 4,
        "A10_minus_A05_min": 2,
        "A20_max_over_A10": 1,
        "object_retained_min": 14,
    }
    for key, value in expected_gate.items():
        if int(frozen.get(key, -1)) != int(value):
            raise ValueError(f"gate.{key} must stay at {value}")
    for key in ("no_errors", "no_nan", "no_object_flew_out", "no_threshold_relaxation"):
        if frozen.get(key) is not True:
            raise ValueError(f"gate.{key} must be true")
    return gate


def label_split(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    labeled: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        init_id = str(row["shared_init_id"])
        if init_id in seen:
            raise ValueError(f"duplicate shared_init_id: {init_id}")
        seen.add(init_id)
        labeled.append(
            dict(row, split="calibration" if init_id in CALIBRATION_INIT_IDS else "confirmation")
        )
    calibration = [row for row in labeled if row["split"] == "calibration"]
    confirmation = [row for row in labeled if row["split"] == "confirmation"]
    if len(labeled) != EXPECTED_INIT_COUNT:
        raise ValueError(f"expected {EXPECTED_INIT_COUNT} rows, got {len(labeled)}")
    if len(calibration) != len(CALIBRATION_INIT_IDS):
        raise ValueError("calibration split is incomplete")
    if len(confirmation) != EXPECTED_CONFIRMATION_COUNT:
        raise ValueError("confirmation split must contain 12 rows")
    return labeled


def _summary_with_extra(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_arm(rows)
    summary["target_not_reached"] = sum(
        row.get("target_destination_not_reached") is True for row in rows
    )
    summary["final_xy_above_0033"] = sum(
        row.get("final_target_xy_m") is not None
        and float(row["final_target_xy_m"]) > 0.033
        for row in rows
    )
    return summary


def _by_split(
    rows: list[dict[str, Any]], arm_id: str
) -> dict[str, dict[str, Any]]:
    arm_rows = [row for row in rows if str(row.get("arm_id")) == arm_id]
    return {
        split: _summary_with_extra(
            [row for row in arm_rows if row.get("split") == split]
        )
        for split in ("calibration", "confirmation")
    }


def evaluate_gate(
    by_arm: Mapping[str, Mapping[str, Any]],
    by_split: Mapping[str, Mapping[str, Mapping[str, Any]]],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    frozen = gate["gate"]
    a10 = by_arm["A10"]
    a10_confirmation = by_split["A10"]["confirmation"]
    clean: dict[str, bool] = {}
    for arm in ARM_ORDER:
        item = by_arm[arm]
        clean[arm] = bool(
            int(item.get("episodes", 0)) == EXPECTED_INIT_COUNT
            and int(item.get("errors", 0)) == 0
            and int(item.get("nan_count", 0)) == 0
            and int(item.get("object_flew_out", 0)) == 0
            and int(item.get("settle_unavailable", 0)) == 0
            and int(item.get("post_done_settle_error_count", 0)) == 0
            and int(item.get("object_retained", 0)) >= int(frozen["object_retained_min"])
        )

    checks = {
        "b0_baseline_reproduced": int(by_arm["B0"]["success"])
        <= int(frozen["B0_success_max"]),
        "a10_total_success": int(a10["success"]) >= int(frozen["A10_success_min"]),
        "a10_confirmation_success": int(a10_confirmation["success"])
        >= int(frozen["A10_confirmation_success_min"]),
        "a10_released_success": int(a10["released_success"])
        >= int(frozen["A10_released_success_min"]),
        "a10_over_a05": int(a10["success"]) - int(by_arm["A05"]["success"])
        >= int(frozen["A10_minus_A05_min"]),
        "a20_overshoot_control": int(by_arm["A20"]["success"])
        <= int(a10["success"]) + int(frozen["A20_max_over_A10"]),
        "stability": all(clean.values()),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if not failed:
        status = "p0_pass"
    elif not checks["b0_baseline_reproduced"]:
        status = "p0_baseline_drift"
    else:
        status = "p0_fail"

    # Final xy over 0.033m is a diagnostic only, not a substitute for the
    # pre-registered success Gate.
    overshoot_counts = {
        arm: int(by_arm[arm].get("final_xy_above_0033", 0)) for arm in ARM_ORDER
    }

    return {
        "status": status,
        "gate_passed": not failed,
        "checks": checks,
        "failure_reasons": failed,
        "diagnostics": {
            "success": {arm: int(by_arm[arm]["success"]) for arm in ARM_ORDER},
            "released_success": {
                arm: int(by_arm[arm]["released_success"]) for arm in ARM_ORDER
            },
            "hold_only_success": {
                arm: int(by_arm[arm]["hold_only_success"]) for arm in ARM_ORDER
            },
            "confirmation_success": {
                arm: int(by_split[arm]["confirmation"]["success"]) for arm in ARM_ORDER
            },
            "calibration_success": {
                arm: int(by_split[arm]["calibration"]["success"]) for arm in ARM_ORDER
            },
            "a10_minus_a05": int(a10["success"]) - int(by_arm["A05"]["success"]),
            "a20_minus_a10": int(by_arm["A20"]["success"]) - int(a10["success"]),
            "final_xy_m": {
                arm: by_arm[arm].get("final_target_xy_m") for arm in ARM_ORDER
            },
            "max_final_xy_above_0033": overshoot_counts,
            "clean": clean,
        },
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# T4b v12 P0 Lateral-Authority + Release/Settle",
        "",
        f"- status: `{summary['status']}`",
        f"- gate_passed: `{summary['gate_passed']}`",
        f"- manifest_sha256: `{summary['manifest_sha256']}`",
        f"- config_sha256: `{summary['gate_config_sha256']}`",
        "",
        "| arm | cap | success | calibration | confirmation | released | hold-only | released placement | wrong destination | target not reached | retained | errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARM_ORDER:
        item = summary["by_arm"][arm]
        split = summary["by_split"][arm]
        lines.append(
            "| {arm} | {cap:.2f} | {success} | {cal} | {conf} | {released} | "
            "{hold} | {placement} | {wrong} | {not_reached} | {retained} | {errors} |".format(
                arm=arm,
                cap=float(EXPECTED_CAPS[arm]),
                success=int(item["success"]),
                cal=int(split["calibration"]["success"]),
                conf=int(split["confirmation"]["success"]),
                released=int(item["released_success"]),
                hold=int(item["hold_only_success"]),
                placement=int(item["released_placement"]),
                wrong=int(item["wrong_destination"]),
                not_reached=int(item["target_not_reached"]),
                retained=int(item["object_retained"]),
                errors=int(item["errors"]),
            )
        )
    lines += [
        "",
        "## Gate",
        "",
        "| check | passed |",
        "|---|---|",
    ]
    for name, passed in summary["checks"].items():
        lines.append(f"| `{name}` | `{str(bool(passed)).lower()}` |")
    if summary.get("failure_reasons"):
        lines += ["", "Failure reasons: " + ", ".join(f"`{x}`" for x in summary["failure_reasons"])]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if args.max_inits not in (0, EXPECTED_INIT_COUNT):
        raise ValueError(f"--max-inits must be 0 or {EXPECTED_INIT_COUNT}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    actual_config_sha = sha256_file(args.gate_config)
    if actual_config_sha != EXPECTED_CONFIG_SHA256:
        raise ValueError(
            "v12 gate config SHA256 mismatch: "
            f"{actual_config_sha} != {EXPECTED_CONFIG_SHA256}"
        )
    manifest_sha = sha256_file(args.manifest)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            f"manifest SHA256 mismatch: {manifest_sha} != {EXPECTED_MANIFEST_SHA256}"
        )

    import torch
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as bddl_utils
    from experiments.robot.libero.run_libero_eval import GenerateConfig

    rows = read_manifest(args.manifest)
    validate_manifest(rows)
    selected = select_white_right(rows, EXPECTED_INIT_COUNT)
    selected = label_split(selected)
    gate = validate_gate(
        json.loads(args.gate_config.read_text()),
        init_count=len({row["shared_init_id"] for row in selected}),
    )
    metric_amendment_sha256 = validate_metric_amendment(gate)
    seed = int(args.seed if args.seed is not None else gate["seed"])
    torch.manual_seed(seed)
    np_random = __import__("numpy").random
    np_random.seed(seed)

    layout = resolve_checkpoint(args.checkpoint)
    fingerprint = checkpoint_fingerprint(layout["fingerprint_files"])
    if fingerprint != EXPECTED_CHECKPOINT_FINGERPRINT:
        raise ValueError(
            f"checkpoint fingerprint mismatch: {fingerprint} != {EXPECTED_CHECKPOINT_FINGERPRINT}"
        )
    if (
        args.expected_checkpoint_fingerprint is not None
        and args.expected_checkpoint_fingerprint != EXPECTED_CHECKPOINT_FINGERPRINT
    ):
        raise ValueError("--expected-checkpoint-fingerprint does not match the frozen value")
    statistics_path = args.statistics or layout.get("statistics")
    if statistics_path is None:
        raise ValueError("--statistics is required for this checkpoint")
    statistics_path = Path(statistics_path)
    if not statistics_path.is_file():
        raise FileNotFoundError(statistics_path)
    statistics_sha = sha256_file(statistics_path)
    if statistics_sha != EXPECTED_STATISTICS_SHA256:
        raise ValueError(
            f"statistics SHA256 mismatch: {statistics_sha} != {EXPECTED_STATISTICS_SHA256}"
        )

    policy, head, proprio, processor, layout, _statistics = load_policy(
        args.model, args.checkpoint, statistics_path
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
        seed=seed,
        env_img_res=args.resolution,
    )
    first = selected[0]
    bddl_path = Path(str(first["bddl_path"]))
    goal_info = goal_info_from_problem(
        bddl_utils.robosuite_parse_problem(str(bddl_path))
    )
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
    by_arm_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARM_ORDER}
    try:
        for arm_id in ARM_ORDER:
            arm_trace_root = trace_root / arm_id
            arm_trace_root.mkdir()
            for row in selected:
                result = run_episode(
                    cfg=cfg,
                    env=env,
                    row=row,
                    policy=policy,
                    head=head,
                    proprio=proprio,
                    processor=processor,
                    gate=gate,
                    arm_id=arm_id,
                    goal_info=goal_info,
                    image_size=args.image_size,
                    use_film=False,
                    seed=seed,
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
                            "fine_cap": EXPECTED_CAPS[arm_id],
                            "row_id": row["row_id"],
                            "split": row.get("split"),
                            "success": result["success"],
                            "released_success": result["released_success"],
                            "hold_only_success": result["hold_only_success"],
                            "released_placement": result["released_placement"],
                            "wrong_destination": result["wrong_destination"],
                            "target_destination_not_reached": result[
                                "target_destination_not_reached"
                            ],
                            "final_target_xy_m": result["final_target_xy_m"],
                            "release_trigger_step": result["release_trigger_step"],
                            "release_action_step": result["release_action_step"],
                            "release_z_before": result["release_z_before"],
                            "object_retained": result["object_retained"],
                            "object_flew_out": result["object_flew_out"],
                            "nan_count": result["nan_count"],
                            "error": result["error"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        env.close()

    by_arm = {arm: _summary_with_extra(by_arm_rows[arm]) for arm in ARM_ORDER}
    by_split = {arm: _by_split(all_rows, arm) for arm in ARM_ORDER}
    decision = evaluate_gate(by_arm, by_split, gate)
    summary = {
        "status": decision["status"],
        "stage": "T4b",
        "audit": "p0_authority_release_v12",
        "note": "Oracle object/plate positions trigger the diagnostic controller only; not a deployable interface.",
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_sha,
        "checkpoint": str(args.checkpoint),
        "checkpoint_format": layout["format"],
        "checkpoint_step": layout["step"],
        "checkpoint_fingerprint": fingerprint,
        "statistics": str(statistics_path),
        "statistics_sha256": statistics_sha,
        "gate_config": str(args.gate_config),
        "gate_config_sha256": actual_config_sha,
        "metric_amendment": gate["metric_amendment"],
        "metric_amendment_sha256": metric_amendment_sha256,
        "init_count": len({row["shared_init_id"] for row in selected}),
        "calibration_init_ids": sorted(CALIBRATION_INIT_IDS),
        "confirmation_init_count": EXPECTED_CONFIRMATION_COUNT,
        "seed": seed,
        "fine_caps": EXPECTED_CAPS,
        "release": {
            "release_xy_m": RELEASE_XY_M,
            "release_z_m": RELEASE_Z_M,
            "release_streak_steps": RELEASE_STREAK_STEPS,
            "settle_steps": SETTLE_STEPS,
        },
        "gate": gate["gate"],
        "checks": decision["checks"],
        "failure_reasons": decision["failure_reasons"],
        "gate_passed": decision["gate_passed"],
        "diagnostics": decision["diagnostics"],
        "by_arm": by_arm,
        "by_split": by_split,
        "episodes": all_rows,
    }
    (args.output / "audit_p0_authority_release_v12.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    (args.output / "audit_p0_authority_release_v12.md").write_text(_markdown(summary))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
