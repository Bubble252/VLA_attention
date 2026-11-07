from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.audit_k4_lite_t4b_v12_p0_authority_release import (
    ARM_ORDER,
    CALIBRATION_INIT_IDS,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_CONFIRMATION_COUNT,
    EXPECTED_INIT_COUNT,
    _summary_with_extra,
    evaluate_gate,
    label_split,
    validate_gate,
)
from scripts.audit_k4_lite_t4b_v8_release_settle import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/vla/k4_lite_t4b_v12_p0_authority_release.json"


def gate_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def arm_summary(
    success: int,
    *,
    released_success: int | None = None,
    hold_only_success: int = 0,
    confirmation_success: int = 0,
    errors: int = 0,
    nan_count: int = 0,
    flew_out: int = 0,
    retained: int = 15,
    settle_unavailable: int = 0,
    final_xy_above_0033: int = 0,
) -> dict:
    return {
        "episodes": EXPECTED_INIT_COUNT,
        "success": success,
        "released_success": success if released_success is None else released_success,
        "hold_only_success": hold_only_success,
        "released_placement": success,
        "target_not_reached": 0,
        "wrong_destination": 0,
        "object_retained": retained,
        "object_flew_out": flew_out,
        "nan_count": nan_count,
        "errors": errors,
        "settle_unavailable": settle_unavailable,
        "post_done_settle_error_count": 0,
        "final_xy_above_0033": final_xy_above_0033,
        "confirmation_success": confirmation_success,
    }


def passing_by_arm() -> dict[str, dict]:
    return {
        "B0": arm_summary(0),
        "A05": arm_summary(1, confirmation_success=0),
        "A10": arm_summary(6, released_success=5, confirmation_success=2),
        "A20": arm_summary(7, released_success=6, confirmation_success=3),
    }


def passing_by_split() -> dict[str, dict[str, dict]]:
    return {
        arm: {
            "calibration": {"success": 0},
            "confirmation": {"success": 0},
        }
        for arm in ARM_ORDER
    } | {
        "A10": {"calibration": {"success": 4}, "confirmation": {"success": 2}},
    }


def test_config_sha_is_frozen() -> None:
    assert sha256_file(CONFIG_PATH) == EXPECTED_CONFIG_SHA256


def test_validate_gate_accepts_frozen_config() -> None:
    gate = validate_gate(gate_config(), init_count=EXPECTED_INIT_COUNT)
    assert tuple(gate["arms"]) == ARM_ORDER
    assert gate["arms"]["B0"]["release_at_target"] is False
    assert gate["arms"]["A10"]["release_xy_m"] == pytest.approx(0.03)
    assert gate["arms"]["A10"]["release_z_m"] == pytest.approx(0.05)
    assert gate["arms"]["A10"]["settle_steps"] == 4


def test_label_split_is_three_calibration_and_twelve_confirmation() -> None:
    ids = sorted(CALIBRATION_INIT_IDS) + [f"confirmation_{index:02d}" for index in range(12)]
    rows = [{"shared_init_id": init_id, "row_id": init_id} for init_id in ids]
    labeled = label_split(rows)
    assert len(labeled) == EXPECTED_INIT_COUNT
    assert sum(row["split"] == "calibration" for row in labeled) == 3
    assert sum(row["split"] == "confirmation" for row in labeled) == EXPECTED_CONFIRMATION_COUNT


def test_label_split_rejects_duplicate_init() -> None:
    rows = [{"shared_init_id": "same", "row_id": str(index)} for index in range(15)]
    with pytest.raises(ValueError, match="duplicate"):
        label_split(rows)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("arms", "A10", "fine_cap"), 0.2, "fine_cap"),
        (("arms", "A10", "release_z_m"), 0.06, "release_z_m"),
        (("arms", "A10", "settle_steps"), 0, "settle_steps"),
        (("arms", "B0", "release_at_target"), True, "must not release"),
        (("metrics", "official_success_xy_m"), 0.04, "official_success_xy_m"),
        (("gate", "A10_released_success_min"), 3, "A10_released_success_min"),
    ],
)
def test_validate_gate_rejects_frozen_field_changes(path: tuple, value, message: str) -> None:
    gate = copy.deepcopy(gate_config())
    node = gate
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        validate_gate(gate, init_count=EXPECTED_INIT_COUNT)


def test_evaluate_gate_pass() -> None:
    decision = evaluate_gate(passing_by_arm(), passing_by_split(), gate_config())
    assert decision["status"] == "p0_pass"
    assert decision["gate_passed"] is True
    assert decision["failure_reasons"] == []


def test_evaluate_gate_rejects_released_success_below_four() -> None:
    by_arm = passing_by_arm()
    by_arm["A10"] = arm_summary(6, released_success=3, confirmation_success=2)
    decision = evaluate_gate(by_arm, passing_by_split(), gate_config())
    assert decision["status"] == "p0_fail"
    assert "a10_released_success" in decision["failure_reasons"]


def test_evaluate_gate_rejects_confirmation_failure() -> None:
    by_split = passing_by_split()
    by_split["A10"]["confirmation"]["success"] = 1
    decision = evaluate_gate(passing_by_arm(), by_split, gate_config())
    assert decision["status"] == "p0_fail"
    assert "a10_confirmation_success" in decision["failure_reasons"]


def test_evaluate_gate_rejects_a20_overshoot() -> None:
    by_arm = passing_by_arm()
    by_arm["A20"] = arm_summary(8, released_success=6, confirmation_success=3)
    decision = evaluate_gate(by_arm, passing_by_split(), gate_config())
    assert decision["status"] == "p0_fail"
    assert "a20_overshoot_control" in decision["failure_reasons"]


def test_evaluate_gate_rejects_nan_or_retention_failure() -> None:
    by_arm = passing_by_arm()
    by_arm["A10"] = arm_summary(
        6,
        released_success=5,
        confirmation_success=2,
        nan_count=1,
        retained=13,
    )
    decision = evaluate_gate(by_arm, passing_by_split(), gate_config())
    assert decision["status"] == "p0_fail"
    assert "stability" in decision["failure_reasons"]


def test_evaluate_gate_marks_baseline_drift_separately() -> None:
    by_arm = passing_by_arm()
    by_arm["B0"] = arm_summary(1)
    decision = evaluate_gate(by_arm, passing_by_split(), gate_config())
    assert decision["status"] == "p0_baseline_drift"
    assert decision["gate_passed"] is False


def test_summary_counts_released_hold_only_and_overshoot() -> None:
    rows = [
        {
            "shared_init_id": "a",
            "success": True,
            "release_action_step": 5,
            "target_destination_not_reached": False,
            "final_target_xy_m": 0.02,
            "released_placement": True,
            "object_retained": True,
            "error": None,
            "nan_count": 0,
        },
        {
            "shared_init_id": "b",
            "success": True,
            "release_action_step": None,
            "target_destination_not_reached": True,
            "final_target_xy_m": 0.04,
            "released_placement": False,
            "object_retained": True,
            "error": None,
            "nan_count": 0,
        },
        {
            "shared_init_id": "c",
            "success": False,
            "release_action_step": None,
            "target_destination_not_reached": True,
            "final_target_xy_m": 0.06,
            "released_placement": False,
            "object_retained": True,
            "error": None,
            "nan_count": 0,
        },
    ]
    summary = _summary_with_extra(rows)
    assert summary["success"] == 2
    assert summary["released_success"] == 1
    assert summary["hold_only_success"] == 1
    assert summary["target_not_reached"] == 2
    assert summary["final_xy_above_0033"] == 2
