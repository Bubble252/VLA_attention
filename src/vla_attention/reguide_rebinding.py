"""ReGuide-style training-free referent rebinding for frozen VLA rollouts.

The wrapper never trains a policy.  It receives a frozen base action and may
replace a bounded prefix with a transport action toward a grounded destination.
For the full arm it then hands control back to the frozen policy and only keeps
lightweight completion/release assistance.

This module is deliberately policy-agnostic.  The caller supplies the grounded
destination position and the simulator/robot observation fields.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass
class RebindState:
    grasp_seen: bool = False
    grasp_step: int | None = None
    activation_step: int | None = None
    transport_active: bool = False
    transport_start_step: int | None = None
    transport_count: int = 0
    handback_requested: bool = False
    handback_step: int | None = None
    monitor_steps: int = 0
    rearm_count: int = 0
    release_active: bool = False
    release_action_step: int | None = None
    release_action_count: int = 0
    settle_complete_step: int | None = None
    release_streak: int = 0
    lost_streak: int = 0
    min_xy_m: float | None = None
    min_z_gap_m: float | None = None
    release_xy_after: float | None = None
    release_z_after: float | None = None
    min_z_gap: float | None = None
    last_applied: str | None = None


def _position(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        position = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if position.size < 3 or not np.all(np.isfinite(position[:3])):
        return None
    return position[:3]


def destination_error(
    object_position: Sequence[float] | None,
    destination_position: Sequence[float] | None,
) -> dict[str, float] | None:
    obj = _position(object_position)
    dst = _position(destination_position)
    if obj is None or dst is None:
        return None
    delta = dst - obj
    return {
        "dx": float(delta[0]),
        "dy": float(delta[1]),
        "dz": float(delta[2]),
        "xy": float(np.hypot(delta[0], delta[1])),
        "xyz": float(np.linalg.norm(delta)),
    }


def in_release_zone(
    object_position: Sequence[float] | None,
    destination_position: Sequence[float] | None,
    *,
    xy_threshold_m: float,
    z_threshold_m: float,
) -> dict[str, Any] | None:
    error = destination_error(object_position, destination_position)
    if error is None:
        return None
    return {
        "in_zone": bool(
            error["xy"] <= float(xy_threshold_m)
            and abs(error["dz"]) <= float(z_threshold_m)
        ),
        "xy_m": error["xy"],
        "z_m": abs(error["dz"]),
        "dx": error["dx"],
        "dy": error["dy"],
        "dz": error["dz"],
    }


def _bounded_component(error: float, *, gain: float, cap: float, deadband: float) -> float:
    if abs(float(error)) <= float(deadband):
        return 0.0
    return float(np.clip(float(gain) * float(error), -float(cap), float(cap)))


def _begin_transport(state: RebindState, control_step: int, *, reason: str) -> None:
    state.transport_active = True
    state.transport_count += 1
    state.transport_start_step = int(control_step)
    state.handback_requested = False
    state.monitor_steps = 0
    state.release_streak = 0
    state.lost_streak = 0
    state.last_applied = reason


def _begin_release(state: RebindState, control_step: int) -> None:
    state.release_active = True
    state.transport_active = False
    if state.release_action_step is None:
        state.release_action_step = int(control_step)


def apply_reguide_control(
    base_action: Any,
    *,
    arm_id: str,
    objects: Mapping[str, Any],
    gripper_width: float | None,
    eef_position: Sequence[float] | None,
    target_object: str,
    destination_name: str,
    destination_position: Sequence[float] | None,
    control_step: int,
    cfg: Mapping[str, Any],
    state: RebindState,
) -> tuple[np.ndarray, dict[str, Any], RebindState]:
    """Apply one ReGuide-style control step.

    ``B0`` is an exact no-op.  ``R-NAIVE`` only servos held-object xy and keeps
    the base z/rotation.  ``R-GEO``/``R-PRED``/``R-SHUF`` use a 3D pre-place
    target, hand control back, then monitor/re-arm/release.
    """
    base = np.asarray(base_action, dtype=np.float32).copy()
    action = base.copy()
    if action.ndim != 1 or action.shape[0] < 7:
        raise ValueError(f"expected a 1-D action with at least 7 dims, got {action.shape}")

    telemetry: dict[str, Any] = {
        "arm_id": arm_id,
        "applied": None,
        "transport_active": bool(state.transport_active),
        "handback_requested": False,
        "release_active": bool(state.release_active),
        "transport_count": int(state.transport_count),
        "transport_start_step": state.transport_start_step,
        "handback_step": state.handback_step,
        "rearm_count": int(state.rearm_count),
        "release_action_step": state.release_action_step,
        "release_action_count": int(state.release_action_count),
        "settle_complete_step": state.settle_complete_step,
        "xy_m": None,
        "z_m": None,
        "dx": None,
        "dy": None,
        "dz": None,
        "action_x_before": float(action[0]),
        "action_y_before": float(action[1]),
        "action_z_before": float(action[2]),
        "action_gripper_before": float(action[6]),
    }
    if arm_id == "B0":
        state.last_applied = None
        telemetry["applied"] = None
        return action, telemetry, state

    destination = _position(destination_position)
    object_pos = _position(objects.get(target_object))
    if destination is None or object_pos is None:
        telemetry["applied"] = "missing_object_or_destination"
        return action, telemetry, state

    zone = in_release_zone(
        object_pos,
        destination,
        xy_threshold_m=float(cfg["release_xy_m"]),
        z_threshold_m=float(cfg["release_z_m"]),
    )
    if zone is not None:
        telemetry.update(
            {
                "xy_m": zone["xy_m"],
                "z_m": zone["z_m"],
                "dx": zone["dx"],
                "dy": zone["dy"],
                "dz": zone["dz"],
            }
        )
        state.min_xy_m = zone["xy_m"] if state.min_xy_m is None else min(state.min_xy_m, zone["xy_m"])
        state.min_z_gap_m = zone["z_m"] if state.min_z_gap_m is None else min(state.min_z_gap_m, zone["z_m"])
        state.min_z_gap = zone["z_m"] if state.min_z_gap is None else min(state.min_z_gap, zone["z_m"])

    if state.release_active:
        action[:6] = 0.0
        action[6] = 1.0
        state.release_action_count += 1
        if state.release_action_count >= int(cfg["settle_steps"]):
            state.settle_complete_step = int(control_step)
        telemetry.update(
            {
                "applied": "release",
                "release_active": True,
                "release_action_count": int(state.release_action_count),
                "settle_complete_step": state.settle_complete_step,
            }
        )
        state.last_applied = "release"
        return action, telemetry, state

    # First grasp is detected from the same width signal used by Goal14.  It is
    # only an execution-state signal; it is not a label and does not read GT.
    if not state.grasp_seen and gripper_width is not None:
        if float(gripper_width) < float(cfg["grasp_gripper_width"]):
            state.grasp_seen = True
            state.grasp_step = int(control_step)

    if not state.transport_active and state.transport_count == 0:
        if state.grasp_seen and state.grasp_step is not None:
            if int(control_step) >= state.grasp_step + int(cfg["activation_delay_steps"]):
                state.activation_step = int(control_step)
                _begin_transport(state, control_step, reason="start_transport")
                telemetry["transport_active"] = True
                telemetry["transport_count"] = int(state.transport_count)
                telemetry["transport_start_step"] = state.transport_start_step
                telemetry["activation_step"] = state.activation_step

    if state.transport_active:
        # The arm-specific target is already materialized by the caller.
        error = destination_error(object_pos, destination)
        if error is None:
            telemetry["applied"] = "missing_error"
            return action, telemetry, state

        target_z = float(destination[2]) + float(cfg.get("approach_height_m", 0.0))
        z_error = target_z - float(object_pos[2])
        action[0] = _bounded_component(
            error["dx"], gain=float(cfg["transport_gain_xy"]), cap=float(cfg["transport_cap_xy"]), deadband=float(cfg["deadband_m"])
        )
        action[1] = _bounded_component(
            error["dy"], gain=float(cfg["transport_gain_xy"]), cap=float(cfg["transport_cap_xy"]), deadband=float(cfg["deadband_m"])
        )
        if arm_id != "R-NAIVE":
            action[2] = _bounded_component(
                z_error, gain=float(cfg["transport_gain_z"]), cap=float(cfg["transport_cap_z"]), deadband=float(cfg["deadband_m"])
            )
        action[6] = 0.0

        if arm_id != "R-NAIVE":
            near_xy = error["xy"] <= float(cfg["handback_xy_m"])
            near_z = abs(z_error) <= float(cfg["handback_z_m"])
            if near_xy and near_z:
                state.transport_active = False
                state.handback_requested = True
                state.handback_step = int(control_step)
                telemetry["handback_requested"] = True
                telemetry["handback_step"] = state.handback_step
                telemetry["applied"] = "handback"
                state.last_applied = "handback"
                return action, telemetry, state
        elif zone is not None and zone["in_zone"]:
            state.release_streak += 1
            if state.release_streak >= int(cfg["release_streak_steps"]):
                _begin_release(state, control_step)
                action[:6] = 0.0
                action[6] = 1.0
                state.release_action_count = 1
                telemetry.update(
                    {
                        "applied": "release",
                        "release_active": True,
                        "release_action_step": state.release_action_step,
                        "release_action_count": int(state.release_action_count),
                    }
                )
                state.last_applied = "release"
                return action, telemetry, state
        else:
            state.release_streak = 0

        if state.transport_start_step is not None:
            if int(control_step) - int(state.transport_start_step) > int(cfg["max_transport_steps"]):
                state.transport_active = False
                if arm_id == "R-NAIVE":
                    telemetry.update(
                        {
                            "applied": "transport_timeout",
                            "transport_active": False,
                            "handback_requested": False,
                            "transport_count": int(state.transport_count),
                            "transport_start_step": state.transport_start_step,
                        }
                    )
                    state.last_applied = "transport_timeout"
                    return base, telemetry, state
                state.handback_requested = True
                state.handback_step = int(control_step)
                telemetry["handback_requested"] = True
                telemetry["handback_step"] = state.handback_step
                telemetry["applied"] = "handback_timeout"
                state.last_applied = "handback_timeout"
                return action, telemetry, state

        telemetry.update(
            {
                "applied": "transport_xy" if arm_id == "R-NAIVE" else "transport_xyz",
                "transport_active": True,
                "transport_count": int(state.transport_count),
                "transport_start_step": state.transport_start_step,
                "action_x_after": float(action[0]),
                "action_y_after": float(action[1]),
                "action_z_after": float(action[2]),
                "action_gripper_after": float(action[6]),
            }
        )
        state.last_applied = telemetry["applied"]
        return action, telemetry, state

    # Before the first grasp/transport activation the wrapper is a strict
    # pass-through.  It is not in the post-hand-back monitor stage yet.
    if state.transport_count == 0:
        telemetry["applied"] = None
        telemetry["transport_active"] = False
        state.last_applied = None
        return action, telemetry, state

    # Hand-back / monitor phase.  The base action is allowed through.  A simple
    # completion certificate releases if the object remains inside the zone,
    # and a lost destination re-arms at most twice.
    state.monitor_steps += 1
    if zone is not None and zone["in_zone"]:
        state.release_streak += 1
        state.lost_streak = 0
        if state.release_streak >= int(cfg["release_streak_steps"]):
            _begin_release(state, control_step)
            action[:6] = 0.0
            action[6] = 1.0
            state.release_action_count = 1
            telemetry.update(
                {
                    "applied": "release_certificate",
                    "release_active": True,
                    "release_action_step": state.release_action_step,
                    "release_action_count": int(state.release_action_count),
                }
            )
            state.last_applied = "release_certificate"
            return action, telemetry, state
    else:
        state.release_streak = 0
        state.lost_streak += 1
        if (
            state.lost_streak >= int(cfg["lost_streak_steps"])
            and state.rearm_count < int(cfg["max_rearms"])
        ):
            state.rearm_count += 1
            _begin_transport(state, control_step, reason="rearm")
            telemetry["applied"] = "rearm"
            telemetry["rearm_count"] = int(state.rearm_count)
            telemetry["transport_active"] = True
            telemetry["transport_count"] = int(state.transport_count)
            telemetry["transport_start_step"] = state.transport_start_step
            state.last_applied = "rearm"
            return action, telemetry, state

    telemetry["applied"] = "policy_handback"
    telemetry["handback_requested"] = False
    telemetry["transport_active"] = False
    telemetry["monitor_steps"] = int(state.monitor_steps)
    state.last_applied = "policy_handback"
    return action, telemetry, state

