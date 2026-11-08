#!/usr/bin/env python3
"""Goal16: ReGuide-style referent rebinding pilot on the frozen C1 policy.

This is a training-free wrapper audit.  It never optimizes a model.  Oracle
destination positions are used only by R-GEO/R-NAIVE/R-SHUF; R-PRED must use
the frozen point-head sidecar and has no oracle fallback.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_k4_lite_t4b_v8_release_settle import (  # noqa: E402
    _append_record,
    _measurement_summary,
    _safe_trace_labels,
    sha256_file,
    summarize_arm,
)
from scripts.eval_c1_counterfactual import read_manifest, validate_manifest  # noqa: E402
from scripts.eval_e1_failure_atlas import (  # noqa: E402
    goal_info_from_problem,
    gripper_width,
    observation_positions,
)
from scripts.eval_j0_rollout import load_policy  # noqa: E402
from scripts.eval_oft_single_checkpoint import (  # noqa: E402
    checkpoint_fingerprint,
    resolve_checkpoint,
)
from src.vla_attention.reguide_rebinding import RebindState, apply_reguide_control  # noqa: E402

CONFIG_VERSION = "goal16_reguide_rebinding_v1"
EXPECTED_CONFIG_SHA256 = "34731e83e56d6c2ca075bdaecbbbb87671379053194b28a4561c06271ea8e502"
EXPECTED_MANIFEST_SHA256 = "e9564ca9aef800e618e3fdb5d144892eabe9471815bbd24bae3140583946f45e"
EXPECTED_CHECKPOINT_FINGERPRINT = "747a28690383ca45e795518121e9aed3f4610265c14618a69f203922b4238e20"
EXPECTED_STATISTICS_SHA256 = "6d5ebb47bc5132df581aae0481d529d04a0821c331859c05a9818d78156c9538"
EXPECTED_POINT_HEAD_SHA256 = "693f0dd8086a4d64cd4f1064a7f4d5c1f934f0914123df5478cdc1f86815fce7"
ARM_ORDER = ("B0", "R-GEO", "R-NAIVE", "R-PRED", "R-SHUF")
CALIBRATION_INIT_IDS = frozenset(
    {
        "init_0e94a4e6b9622c6b",
        "init_2edf49999eb35566",
        "init_3500df19a4d3b5a1",
    }
)
EXPECTED_INIT_COUNT = 15
EXPECTED_CONFIRMATION_COUNT = 12
HAND_BACK_APPLIED = frozenset(
    {"handback", "handback_timeout", "policy_handback", "rearm"}
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gate-config", type=Path, required=True)
    parser.add_argument("--grounding-sidecar", type=Path, default=None)
    parser.add_argument("--split-audit", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", default=",".join(ARM_ORDER))
    parser.add_argument("--max-inits", type=int, default=0)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-open-loop-steps", type=int, default=8)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--center-crop", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _require_number(cfg: Mapping[str, Any], key: str, expected: float, *, label: str) -> None:
    if key not in cfg:
        raise ValueError(f"missing {label}.{key}")
    actual = float(cfg[key])
    if abs(actual - float(expected)) > 1e-12:
        raise ValueError(f"{label}.{key} must be {expected}, got {actual}")


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("version") != CONFIG_VERSION:
        raise ValueError(f"unexpected config version: {config.get('version')!r}")
    if config.get("status") != "frozen_before_measurement":
        raise ValueError("config is not frozen")
    if int(config.get("init_count", -1)) != EXPECTED_INIT_COUNT:
        raise ValueError("init_count must be 15")
    if int(config.get("confirmation_init_count", -1)) != EXPECTED_CONFIRMATION_COUNT:
        raise ValueError("confirmation_init_count must be 12")
    if set(config.get("calibration_init_ids") or []) != set(CALIBRATION_INIT_IDS):
        raise ValueError("calibration_init_ids changed")
    if config.get("target_object") != "porcelain_mug_1":
        raise ValueError("target_object must be porcelain_mug_1")
    if config.get("target_receptacle") != "plate_2":
        raise ValueError("target_receptacle must be plate_2")
    if config.get("distractor_receptacle") != "plate_1":
        raise ValueError("distractor_receptacle must be plate_1")
    if int(config.get("seed", -1)) != 7:
        raise ValueError("seed must be 7")
    if tuple(config.get("arms") or {}) != ARM_ORDER:
        raise ValueError(f"arms must be exactly {ARM_ORDER}")
    for arm_id in ARM_ORDER:
        arm = config["arms"][arm_id]
        if arm_id == "B0" and arm.get("kind") != "noop":
            raise ValueError("B0 must be noop")
        if arm_id != "B0" and arm.get("kind") not in {"full", "naive"}:
            raise ValueError(f"unexpected kind for {arm_id}: {arm.get('kind')!r}")
    grounding = config.get("grounding") or {}
    if grounding.get("oracle_fallback_allowed") is not False:
        raise ValueError("oracle_fallback_allowed must be false")
    if grounding.get("point_head_sha256") != EXPECTED_POINT_HEAD_SHA256:
        raise ValueError("point_head_sha256 changed")
    release = config.get("release") or {}
    for key, value in (
        ("release_xy_m", 0.03),
        ("release_z_m", 0.05),
        ("release_streak_steps", 3),
        ("settle_steps", 4),
    ):
        _require_number(release, key, value, label="release")
    shared = config.get("shared_control") or {}
    for key, value in (
        ("grasp_gripper_width", 0.04),
        ("activation_delay_steps", 2),
        ("deadband_m", 0.003),
        ("transport_gain_xy", 2.0),
        ("transport_cap_xy", 0.2),
        ("transport_gain_z", 2.0),
        ("transport_cap_z", 0.15),
        ("max_transport_steps", 240),
        ("lost_streak_steps", 3),
        ("max_rearms", 2),
    ):
        _require_number(shared, key, value, label="shared_control")
    for arm_id in ("R-GEO", "R-PRED", "R-SHUF"):
        arm = config["arms"][arm_id]
        if abs(float(arm.get("approach_height_m", -1)) - 0.10) > 1e-12:
            raise ValueError(f"{arm_id}.approach_height_m must be 0.10")
        if abs(float(arm.get("handback_z_m", -1)) - 0.06) > 1e-12:
            raise ValueError(f"{arm_id}.handback_z_m must be 0.06")
    if abs(float(config["arms"]["R-NAIVE"].get("approach_height_m", -1))) > 1e-12:
        raise ValueError("R-NAIVE.approach_height_m must be 0")
    return config


def label_split(rows: Sequence[dict[str, Any]], *, allow_partial: bool) -> list[dict[str, Any]]:
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
    if not allow_partial:
        calibration = [row for row in labeled if row["split"] == "calibration"]
        confirmation = [row for row in labeled if row["split"] == "confirmation"]
        if len(labeled) != EXPECTED_INIT_COUNT:
            raise ValueError(f"expected {EXPECTED_INIT_COUNT} rows, got {len(labeled)}")
        if len(calibration) != len(CALIBRATION_INIT_IDS):
            raise ValueError("calibration split is incomplete")
        if len(confirmation) != EXPECTED_CONFIRMATION_COUNT:
            raise ValueError("confirmation split must contain 12 rows")
    return labeled


def select_rows(rows: list[dict[str, Any]], max_inits: int) -> list[dict[str, Any]]:
    selected = sorted(
        [dict(row) for row in rows if str(row.get("prompt_id")) == "white_right"],
        key=lambda row: str(row["shared_init_id"]),
    )
    if max_inits > 0:
        selected = selected[:max_inits]
    if not selected:
        raise ValueError("no white_right rows")
    return selected


def _control_cfg(config: Mapping[str, Any], arm_id: str) -> dict[str, Any]:
    shared = dict(config["shared_control"])
    release = dict(config["release"])
    arm = dict(config["arms"][arm_id])
    return {
        **shared,
        "release_xy_m": float(release["release_xy_m"]),
        "release_z_m": float(release["release_z_m"]),
        "release_streak_steps": int(release["release_streak_steps"]),
        "settle_steps": int(release["settle_steps"]),
        "approach_height_m": float(arm.get("approach_height_m", 0.0)),
        "handback_xy_m": float(arm.get("handback_xy_m", 0.0)),
        "handback_z_m": float(arm.get("handback_z_m", 0.0)),
    }


def _grounding_rows(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {str(row["shared_init_id"]): row for row in rows}


def _destination_for(
    arm_id: str,
    *,
    obs: Mapping[str, Any],
    config: Mapping[str, Any],
    grounding: Mapping[str, Any] | None,
) -> tuple[np.ndarray | None, str]:
    if arm_id in {"R-GEO", "R-NAIVE"}:
        name = str(config["target_receptacle"])
        value = obs.get(f"{name}_pos")
    elif arm_id == "R-SHUF":
        name = str(config["distractor_receptacle"])
        value = obs.get(f"{name}_pos")
    elif arm_id == "R-PRED":
        if grounding is None or grounding.get("destination_position_xyz") is None:
            return None, "predicted_missing"
        return np.asarray(grounding["destination_position_xyz"], dtype=np.float64), "predicted"
    else:
        return None, "none"
    if value is None:
        return None, name
    return np.asarray(value, dtype=np.float64)[:3], name


def run_episode(
    *,
    cfg: Any,
    env: Any,
    row: dict[str, Any],
    policy: Any,
    head: Any,
    proprio: Any,
    processor: Any,
    config: Mapping[str, Any],
    arm_id: str,
    goal_info: dict[str, Any],
    image_size: int,
    use_film: bool,
    grounding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    import experiments.robot.libero.run_libero_eval as official

    target = str(config["target_object"])
    metrics_cfg = config["metrics"]
    control_cfg = _control_cfg(config, arm_id)
    env.reset()
    initial_state = np.asarray(row["init_state"], dtype=np.float64).copy()
    if row.get("sim_time_set_to_zero", False):
        initial_state[0] = 0.0
    obs = env.set_init_state(initial_state)
    for _ in range(int(cfg.num_steps_wait)):
        obs, _reward, _done, _info = env.step(official.get_libero_dummy_action(cfg.model_family))

    action_queue: deque[Any] = deque(maxlen=cfg.num_open_loop_steps)
    records: list[dict[str, Any]] = []
    max_steps = official.TASK_MAX_STEPS[official.TaskSuite(cfg.task_suite_name)]
    success = False
    success_step: int | None = None
    control_step = 0
    error = None
    state = RebindState()
    action_change_max = 0.0
    x_change_max = 0.0
    y_change_max = 0.0
    z_change_max = 0.0
    gripper_change_max = 0.0
    destination_source = "none"

    objects = observation_positions(obs)
    records.append(
        {
            "step": 0,
            "control_step": 0,
            "phase": "baseline",
            "eef_pos": [float(x) for x in obs.get("robot0_eef_pos", [])]
            if obs.get("robot0_eef_pos") is not None
            else None,
            "eef_quat": [float(x) for x in obs.get("robot0_eef_quat", [])]
            if obs.get("robot0_eef_quat") is not None
            else None,
            "gripper_width": gripper_width(obs),
            "objects": objects,
        }
    )

    try:
        while control_step < int(max_steps):
            if len(action_queue) == 0:
                observation, _image = official.prepare_observation(obs, image_size)
                actions = official.get_action(
                    cfg,
                    policy,
                    observation,
                    row["prompt_language"],
                    processor=processor,
                    action_head=head,
                    proprio_projector=proprio,
                    noisy_action_projector=None,
                    use_film=use_film,
                )
                action_queue.extend(np.asarray(actions, dtype=np.float32))

            base_action = np.asarray(action_queue.popleft(), dtype=np.float32)
            destination, destination_source = _destination_for(
                arm_id, obs=obs, config=config, grounding=grounding
            )
            model_action, telemetry, state = apply_reguide_control(
                base_action,
                arm_id=arm_id,
                objects=observation_positions(obs),
                gripper_width=gripper_width(obs),
                eef_position=obs.get("robot0_eef_pos"),
                target_object=target,
                destination_name=destination_source,
                destination_position=destination,
                control_step=control_step,
                cfg=control_cfg,
                state=state,
            )
            action_diff = np.abs(np.asarray(model_action) - base_action)
            action_change_max = max(action_change_max, float(action_diff.max()))
            x_change_max = max(x_change_max, float(action_diff[0]))
            y_change_max = max(y_change_max, float(action_diff[1]))
            z_change_max = max(z_change_max, float(action_diff[2]))
            gripper_change_max = max(gripper_change_max, float(action_diff[6]))
            processed = official.process_action(model_action, cfg.model_family)
            obs, reward, done, _info = env.step(processed.tolist())
            _append_record(
                records,
                obs=obs,
                base_action=base_action,
                model_action=model_action,
                processed=processed,
                reward=float(reward),
                done=bool(done),
                arm_id=arm_id,
                control_step=control_step,
                total_step=control_step + int(cfg.num_steps_wait),
                phase="policy",
                telemetry=telemetry,
                state=state,
                target=target,
                target_receptacle=str(config["target_receptacle"]),
                arm_cfg=control_cfg,
                metrics_cfg=metrics_cfg,
            )
            if telemetry.get("applied") in HAND_BACK_APPLIED or telemetry.get(
                "handback_requested"
            ):
                action_queue.clear()
            control_step += 1
            if done:
                success = True
                success_step = control_step - 1
                break
            if state.settle_complete_step is not None:
                break
    except Exception as exc:  # pragma: no cover - remote environment
        error = repr(exc)

    labels, destination, destination_label = _safe_trace_labels(
        records=records,
        goal_info=goal_info,
        target=target,
        target_receptacle=str(config["target_receptacle"]),
        success=success,
        threshold=float(metrics_cfg["placement_xy_m"]),
    )
    measurements = _measurement_summary(
        records,
        target=target,
        target_receptacle=str(config["target_receptacle"]),
        metrics_cfg=metrics_cfg,
        release_action_step=state.release_action_step,
        settle_complete_step=state.settle_complete_step,
    )
    released_success = bool(success and state.release_action_step is not None)
    return {
        **row,
        "arm_id": arm_id,
        "success": bool(success),
        "official_success_step": success_step,
        "released_success": released_success,
        "hold_only_success": bool(success and state.release_action_step is None),
        "steps": int(control_step),
        "evaluation_complete": bool(error is None),
        "error": error,
        "destination_source": destination_source,
        "transport_started": bool(state.transport_count > 0),
        "transport_count": int(state.transport_count),
        "transport_start_step": state.transport_start_step,
        "handback_seen": state.handback_step is not None,
        "handback_step": state.handback_step,
        "rearm_count": int(state.rearm_count),
        "release_action_step": state.release_action_step,
        "release_action_count": int(state.release_action_count),
        "settle_complete_step": state.settle_complete_step,
        "min_xy_m": state.min_xy_m,
        "min_z_gap_m": state.min_z_gap_m,
        "action_change_max": action_change_max,
        "x_action_change_max": x_change_max,
        "y_action_change_max": y_change_max,
        "z_action_change_max": z_change_max,
        "gripper_action_change_max": gripper_change_max,
        **labels,
        **destination,
        **destination_label,
        **measurements,
        "records": records,
    }


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


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _summary_with_extra(rows)
    summary["transport_started"] = sum(row.get("transport_started") is True for row in rows)
    summary["handback_seen"] = sum(row.get("handback_seen") is True for row in rows)
    summary["rearm_total"] = sum(int(row.get("rearm_count", 0)) for row in rows)
    summary["evaluation_complete"] = sum(
        row.get("evaluation_complete") is True for row in rows
    )
    return summary


def _by_split(rows: list[dict[str, Any]], arm_id: str) -> dict[str, dict[str, Any]]:
    arm_rows = [row for row in rows if str(row.get("arm_id")) == arm_id]
    return {
        split: _summary([row for row in arm_rows if row.get("split") == split])
        for split in ("calibration", "confirmation")
    }


def evaluate_gate(
    by_arm: Mapping[str, Mapping[str, Any]],
    *,
    arms: Sequence[str],
    init_count: int,
    config: Mapping[str, Any],
    grounding_status: Mapping[str, Any],
) -> dict[str, Any]:
    gate = config["gate"]
    checks: dict[str, bool] = {}
    for arm in arms:
        item = by_arm.get(arm, {})
        checks[f"{arm}_rows"] = int(item.get("episodes", 0)) == int(init_count)
        checks[f"{arm}_clean"] = bool(
            int(item.get("errors", 0)) == 0
            and int(item.get("nan_count", 0)) == 0
            and int(item.get("object_flew_out", 0)) == 0
        )
    checks["g1_geo_handback"] = (
        int(by_arm["R-GEO"].get("transport_started", 0)) >= int(gate["g1_geo_handback_min"])
        and int(by_arm["R-GEO"].get("handback_seen", 0)) >= int(gate["g1_geo_handback_min"])
    ) if "R-GEO" in by_arm else False
    checks["g1_shuf_not_better_than_b0"] = (
        int(by_arm["R-SHUF"].get("success", 0)) <= int(by_arm["B0"].get("success", 0))
    ) if "B0" in by_arm and "R-SHUF" in by_arm else True
    checks["g2_r_geo_success"] = (
        int(by_arm["R-GEO"].get("success", 0)) >= int(gate["g2_r_geo_success_min"])
    ) if "R-GEO" in by_arm else False
    checks["g2_r_geo_released_success"] = (
        int(by_arm["R-GEO"].get("released_success", 0))
        >= int(gate["g2_r_geo_released_success_min"])
    ) if "R-GEO" in by_arm else False
    checks["g2_r_pred_grounding_valid"] = bool(grounding_status.get("passed") is True)
    checks["g2_r_pred_success"] = (
        bool(grounding_status.get("passed") is True)
        and int(by_arm.get("R-PRED", {}).get("success", 0))
        >= int(gate["g2_r_pred_success_min"])
    )
    checks["g2_r_pred_minus_b0"] = (
        bool(grounding_status.get("passed") is True)
        and int(by_arm.get("R-PRED", {}).get("success", 0))
        - int(by_arm.get("B0", {}).get("success", 0))
        >= int(gate["g2_r_pred_minus_b0_min"])
    )
    checks["g2_r_geo_wrong_destination"] = (
        int(by_arm["R-GEO"].get("wrong_destination", 0))
        <= int(by_arm["B0"].get("wrong_destination", 0))
        + int(gate["g2_wrong_destination_max_over_b0"])
    ) if "R-GEO" in by_arm and "B0" in by_arm else False
    g1_names = [name for name in checks if name.startswith("g1_") or name.endswith("_rows") or name.endswith("_clean")]
    g1_passed = all(checks[name] for name in g1_names)
    g2_passed = all(checks[name] for name in checks if name.startswith("g2_"))
    if not g1_passed:
        status = "INVALID" if not checks.get("g2_r_pred_grounding_valid", True) else "FAIL"
    elif not g2_passed:
        status = "INVALID" if not checks.get("g2_r_pred_grounding_valid", True) else "FAIL"
    else:
        status = "PASS"
    return {
        "status": status,
        "g1_passed": g1_passed,
        "g2_passed": g2_passed,
        "checks": checks,
        "failed": [name for name, passed in checks.items() if not passed],
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Goal16 ReGuide-style rebinding pilot",
        "",
        f"- status: `{summary['status']}`",
        f"- G1 passed: `{summary['gate']['g1_passed']}`",
        f"- G2 passed: `{summary['gate']['g2_passed']}`",
        f"- manifest SHA256: `{summary['manifest_sha256']}`",
        f"- checkpoint fingerprint: `{summary['checkpoint_fingerprint']}`",
        f"- statistics SHA256: `{summary['statistics_sha256']}`",
        f"- config SHA256: `{summary['config_sha256']}`",
        f"- grounding status: `{summary['grounding']['status']}`",
        "",
        "## Arms",
        "",
        "| arm | episodes | success | released | hold-only | transport | hand-back | wrong dest | errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in summary["arms_order"]:
        item = summary["by_arm"][arm]
        lines.append(
            f"| `{arm}` | {item.get('episodes')} | {item.get('success')} | "
            f"{item.get('released_success')} | {item.get('hold_only_success')} | "
            f"{item.get('transport_started')} | {item.get('handback_seen')} | "
            f"{item.get('wrong_destination')} | {item.get('errors')} |"
        )
    lines += ["", "## Failed checks", ""]
    failed = summary["gate"]["failed"]
    lines += [f"- `{name}`" for name in failed] if failed else ["- none"]
    lines += ["", "## Grounding", "", "```json", json.dumps(summary["grounding"], indent=2), "```", ""]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config_sha = sha256_file(args.gate_config)
    if config_sha != EXPECTED_CONFIG_SHA256:
        raise ValueError(f"config SHA256 mismatch: {config_sha} != {EXPECTED_CONFIG_SHA256}")
    manifest_sha = sha256_file(args.manifest)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"manifest SHA256 mismatch: {manifest_sha}")
    config = validate_config(json.loads(args.gate_config.read_text()))
    selected_arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    if any(arm not in ARM_ORDER for arm in selected_arms):
        raise ValueError(f"unknown arm in {selected_arms}")
    if not args.allow_partial and selected_arms != list(ARM_ORDER):
        raise ValueError("official run must use all five arms in order")

    import torch
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as bddl_utils
    from experiments.robot.libero.run_libero_eval import GenerateConfig

    rows = read_manifest(args.manifest)
    validate_manifest(rows)
    selected = label_split(
        select_rows(rows, args.max_inits), allow_partial=args.allow_partial
    )
    if not args.allow_partial and len(selected) != EXPECTED_INIT_COUNT:
        raise ValueError(f"expected {EXPECTED_INIT_COUNT} rows, got {len(selected)}")

    split_audit = None
    if args.split_audit is not None:
        split_audit = json.loads(args.split_audit.read_text())
    init_shas = {str(row["init_state_sha256"]) for row in selected}
    split_ok = True
    if split_audit is not None and not args.allow_partial:
        split_ok = bool(
            split_audit.get("status") == "PASS"
            and split_audit.get("gates", {}).get("white_right_train_zero") is True
            and split_audit.get("gates", {}).get("p0_eval_disjoint_from_train") is True
            and split_audit.get("gates", {}).get("p0_eval_disjoint_from_validation") is True
            and set(split_audit.get("p0_white_right", {}).get("init_state_shas", [])) == init_shas
        )

    grounding_status: dict[str, Any] = {"status": "NOT_USED", "passed": None}
    grounding_rows: dict[str, dict[str, Any]] = {}
    if args.grounding_sidecar is not None:
        grounding_rows = _grounding_rows(args.grounding_sidecar)
        summary_path = args.grounding_sidecar.parent / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text())
        actual_sidecar_sha = sha256_file(args.grounding_sidecar)
        if summary.get("sidecar_sha256") != actual_sidecar_sha:
            raise ValueError("grounding sidecar SHA mismatch")
        grounding_status = {
            "status": "PASS" if summary.get("passed") is True else "INVALID",
            "passed": bool(summary.get("passed") is True),
            "summary": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "sidecar": str(args.grounding_sidecar),
            "sidecar_sha256": actual_sidecar_sha,
            "max_destination_error_m": summary.get("max_destination_error_m"),
            "grounding_miss_count": summary.get("grounding_miss_count"),
            "selected_init_count": summary.get("selected_init_count"),
            "point_head_sha256": summary.get("point_head_sha256"),
        }
        if grounding_status["point_head_sha256"] != EXPECTED_POINT_HEAD_SHA256:
            raise ValueError("grounding sidecar point head SHA mismatch")
        if not args.allow_partial and set(grounding_rows) != {
            str(row["shared_init_id"]) for row in selected
        }:
            raise ValueError("grounding sidecar does not cover exactly the selected inits")
        if not args.allow_partial and summary.get("selected_init_count") != EXPECTED_INIT_COUNT:
            raise ValueError("grounding sidecar must cover 15 inits")

    seed = int(args.seed if args.seed is not None else config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    layout = resolve_checkpoint(args.checkpoint)
    fingerprint = checkpoint_fingerprint(layout["fingerprint_files"])
    if fingerprint != EXPECTED_CHECKPOINT_FINGERPRINT:
        raise ValueError(f"checkpoint fingerprint mismatch: {fingerprint}")
    statistics_sha = sha256_file(args.statistics)
    if statistics_sha != EXPECTED_STATISTICS_SHA256:
        raise ValueError(f"statistics SHA256 mismatch: {statistics_sha}")

    policy, head, proprio, processor, layout, _statistics = load_policy(
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
        seed=seed,
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
                result = run_episode(
                    cfg=cfg,
                    env=env,
                    row=row,
                    policy=policy,
                    head=head,
                    proprio=proprio,
                    processor=processor,
                    config=config,
                    arm_id=arm_id,
                    goal_info=goal_info,
                    image_size=args.image_size,
                    use_film=False,
                    grounding=grounding_rows.get(str(row["shared_init_id"])),
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
                            "split": row.get("split"),
                            "success": result["success"],
                            "released_success": result["released_success"],
                            "transport_started": result["transport_started"],
                            "handback_seen": result["handback_seen"],
                            "wrong_destination": result["wrong_destination"],
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
    gate_result = evaluate_gate(
        by_arm,
        arms=selected_arms,
        init_count=len(selected),
        config=config,
        grounding_status=grounding_status,
    ) if not args.allow_partial else {"status": "SMOKE", "g1_passed": None, "g2_passed": None, "checks": {}, "failed": []}
    if not args.allow_partial:
        gate_result["checks"]["split_contract"] = bool(split_ok)
        if not split_ok and gate_result["status"] != "INVALID":
            gate_result["status"] = "INVALID"
            gate_result["failed"] = list(gate_result.get("failed", [])) + ["split_contract"]
    summary = {
        "status": gate_result["status"],
        "stage": "Goal16",
        "audit": "reguide_rebinding_pilot",
        "note": "Training-free wrapper; oracle destinations never used by R-PRED.",
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_sha,
        "checkpoint": str(args.checkpoint),
        "checkpoint_format": layout["format"],
        "checkpoint_step": layout["step"],
        "checkpoint_fingerprint": fingerprint,
        "statistics": str(args.statistics),
        "statistics_sha256": statistics_sha,
        "config": str(args.gate_config),
        "config_sha256": config_sha,
        "split_audit": None if args.split_audit is None else str(args.split_audit),
        "split_audit_sha256": None if args.split_audit is None else sha256_file(args.split_audit),
        "split_ok": split_ok,
        "init_count": len(selected),
        "calibration_init_ids": sorted(CALIBRATION_INIT_IDS),
        "seed": seed,
        "arms_order": selected_arms,
        "grounding": grounding_status,
        "gate": gate_result,
        "by_arm": by_arm,
        "by_split": by_split,
        "episodes": all_rows,
        "elapsed_seconds": time.time() - started,
    }
    (args.output / "audit_goal16_reguide_rebinding.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    (args.output / "audit_goal16_reguide_rebinding.md").write_text(_markdown(summary))
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2, ensure_ascii=False))
    return 0 if summary["status"] in {"PASS", "SMOKE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
