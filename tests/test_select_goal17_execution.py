from __future__ import annotations

from scripts.select_goal17_execution import evaluate_arm


def _gate() -> dict:
    return {
        "grounding_init_denominator": 15,
        "execution_r_pred_success_min": 1,
        "execution_r_pred_minus_b0_min": 1,
        "execution_released_success_min": 1,
    }


def _metrics(success: int = 2) -> dict:
    return {
        "episodes": 15,
        "success": success,
        "released_success": 2,
        "wrong_destination": 3,
        "errors": 0,
        "nan_count": 0,
        "object_flew_out": 0,
    }


def test_goal17_execution_gate_passes_with_positive_arm() -> None:
    result = evaluate_arm(
        _metrics(),
        b0={"success": 0, "wrong_destination": 14},
        gate=_gate(),
        grounding_passed=True,
    )
    assert result["passed"] is True
    assert result["failed"] == []


def test_goal17_execution_gate_rejects_zero_success_and_bad_grounding() -> None:
    result = evaluate_arm(
        _metrics(success=0),
        b0={"success": 0, "wrong_destination": 14},
        gate=_gate(),
        grounding_passed=False,
    )
    assert result["passed"] is False
    assert "success" in result["failed"]
    assert "success_minus_b0" in result["failed"]
    assert "grounding_passed" in result["failed"]
