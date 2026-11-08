from __future__ import annotations

import numpy as np

from src.vla_attention.reguide_rebinding import (
    RebindState,
    apply_reguide_control,
    destination_error,
    in_release_zone,
)


def _cfg() -> dict:
    return {
        "grasp_gripper_width": 0.04,
        "activation_delay_steps": 2,
        "deadband_m": 0.003,
        "transport_gain_xy": 2.0,
        "transport_cap_xy": 0.2,
        "transport_gain_z": 2.0,
        "transport_cap_z": 0.15,
        "max_transport_steps": 10,
        "release_xy_m": 0.03,
        "release_z_m": 0.05,
        "release_streak_steps": 3,
        "settle_steps": 4,
        "lost_streak_steps": 3,
        "max_rearms": 2,
        "approach_height_m": 0.10,
        "handback_xy_m": 0.05,
        "handback_z_m": 0.06,
    }


def _call(
    base_action,
    *,
    arm_id,
    objects,
    destination,
    control_step,
    state,
    gripper_width=0.03,
):
    return apply_reguide_control(
        np.asarray(base_action, dtype=np.float32),
        arm_id=arm_id,
        objects=objects,
        gripper_width=gripper_width,
        eef_position=[0.0, 0.0, 0.5],
        target_object="target",
        destination_name="dest",
        destination_position=destination,
        control_step=control_step,
        cfg=_cfg(),
        state=state,
    )


def test_b0_is_exact_noop() -> None:
    base = np.array([0.1, -0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
    state = RebindState()
    action, telemetry, returned = _call(
        base,
        arm_id="B0",
        objects={"target": [0.0, 0.0, 0.4]},
        destination=[0.1, 0.1, 0.44],
        control_step=0,
        state=state,
    )
    assert np.array_equal(action, base)
    assert telemetry["applied"] is None
    assert returned.transport_count == 0
    assert returned.release_active is False


def test_destination_error_and_release_zone_are_xyz_aware() -> None:
    error = destination_error([0.0, 0.0, 0.4], [0.03, 0.04, 0.45])
    assert error is not None
    assert np.isclose(error["xy"], 0.05)
    assert np.isclose(error["dz"], 0.05)
    zone = in_release_zone(
        [0.0, 0.0, 0.4],
        [0.03, 0.04, 0.45],
        xy_threshold_m=0.05,
        z_threshold_m=0.05,
    )
    assert zone is not None and zone["in_zone"] is True


def test_geo_transports_xyz_then_hands_back_and_settles() -> None:
    cfg = _cfg()
    state = RebindState()
    base = np.zeros(7, dtype=np.float32)
    base[6] = -1.0
    objects = {"target": [0.0, 0.0, 0.4], "dest": [0.1, 0.1, 0.44]}

    _, warm, state = _call(
        base,
        arm_id="R-GEO",
        objects=objects,
        destination=objects["dest"],
        control_step=0,
        state=state,
    )
    assert warm["applied"] is None

    action, start, state = _call(
        base,
        arm_id="R-GEO",
        objects=objects,
        destination=objects["dest"],
        control_step=2,
        state=state,
    )
    assert start["applied"] == "transport_xyz"
    assert action[0] > 0 and action[1] > 0 and action[2] > 0
    assert action[6] == 0.0

    objects["target"] = [0.1, 0.1, 0.54]
    _, handback, state = _call(
        base,
        arm_id="R-GEO",
        objects=objects,
        destination=objects["dest"],
        control_step=3,
        state=state,
    )
    assert handback["handback_requested"] is True
    assert handback["applied"] == "handback"

    # The frozen policy takes control after hand-back and lowers the object.
    objects["target"] = [0.1, 0.1, 0.44]
    applied = []
    for step in range(4, 12):
        action, telemetry, state = _call(
            base,
            arm_id="R-GEO",
            objects=objects,
            destination=objects["dest"],
            control_step=step,
            state=state,
        )
        applied.append(telemetry["applied"])
    assert "release_certificate" in applied
    assert state.release_active is True
    assert state.release_action_count >= cfg["settle_steps"]
    assert state.settle_complete_step is not None


def test_naive_only_overrides_xy_and_never_requests_handback() -> None:
    state = RebindState()
    base = np.array([0.0, 0.0, 0.25, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    objects = {"target": [0.0, 0.0, 0.4], "dest": [0.1, 0.1, 0.44]}
    _call(
        base,
        arm_id="R-NAIVE",
        objects=objects,
        destination=objects["dest"],
        control_step=0,
        state=state,
    )
    action, telemetry, state = _call(
        base,
        arm_id="R-NAIVE",
        objects=objects,
        destination=objects["dest"],
        control_step=2,
        state=state,
    )
    assert telemetry["applied"] == "transport_xy"
    assert action[2] == base[2]
    assert action[0] != base[0] and action[1] != base[1]

    _, timeout, state = _call(
        base,
        arm_id="R-NAIVE",
        objects={"target": [0.0, 0.0, 0.4], "dest": [1.0, 1.0, 0.44]},
        destination=[1.0, 1.0, 0.44],
        control_step=2 + _cfg()["max_transport_steps"] + 1,
        state=state,
    )
    assert timeout["applied"] == "transport_timeout"
    assert timeout["handback_requested"] is False
    assert state.handback_step is None
    assert state.rearm_count == 0


def test_naive_releases_inside_zone_without_handback() -> None:
    state = RebindState()
    base = np.zeros(7, dtype=np.float32)
    base[6] = -1.0
    objects = {"target": [0.0, 0.0, 0.44], "dest": [0.0, 0.0, 0.44]}
    _call(
        base,
        arm_id="R-NAIVE",
        objects=objects,
        destination=objects["dest"],
        control_step=0,
        state=state,
    )
    telemetry = None
    for step in range(2, 10):
        _, telemetry, state = _call(
            base,
            arm_id="R-NAIVE",
            objects=objects,
            destination=objects["dest"],
            control_step=step,
            state=state,
        )
    assert telemetry is not None and telemetry["applied"] == "release"
    assert state.release_active is True
    assert state.handback_step is None


def test_shuffled_destination_changes_transport_direction() -> None:
    base = np.zeros(7, dtype=np.float32)
    base[6] = -1.0
    state = RebindState()
    objects = {"target": [0.0, 0.0, 0.4]}
    _, left, _ = _call(
        base,
        arm_id="R-SHUF",
        objects=objects,
        destination=[-0.2, -0.1, 0.44],
        control_step=0,
        state=state,
    )
    _, _, _ = _call(
        base,
        arm_id="R-SHUF",
        objects=objects,
        destination=[-0.2, -0.1, 0.44],
        control_step=2,
        state=state,
    )
    state = RebindState()
    _, _, _ = _call(
        base,
        arm_id="R-SHUF",
        objects=objects,
        destination=[-0.2, -0.1, 0.44],
        control_step=0,
        state=state,
    )
    right_action, right, _ = _call(
        base,
        arm_id="R-SHUF",
        objects=objects,
        destination=[0.2, 0.1, 0.44],
        control_step=2,
        state=state,
    )
    assert left["applied"] is None and right["applied"] == "transport_xyz"
    assert right_action[0] > 0 and right_action[1] > 0
