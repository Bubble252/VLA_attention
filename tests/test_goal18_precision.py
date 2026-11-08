from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_goal17_grounding import policy_point_to_raw
from scripts.evaluate_goal18_grounding import (
    evaluate_gate,
    offset_for,
    raw_to_policy_point,
)
from src.vla_attention.destination_residual import (
    DestinationResidualHead,
    boundary_features,
    normalize_features,
)
from scripts.audit_goal18_execution import evaluate_goal18_gate


def test_policy_raw_roundtrip() -> None:
    policy = (91.5, 104.25)
    raw = policy_point_to_raw(
        policy,
        image_size=224,
        raw_resolution=256,
        crop_scale=0.9,
    )
    restored = raw_to_policy_point(
        raw,
        image_size=224,
        raw_resolution=256,
        crop_scale=0.9,
    )
    assert np.allclose(restored, policy, atol=1e-6)


def test_offset_lookup() -> None:
    offsets = {
        "global": [0.1, -0.2],
        "per_prompt": {"white_right": [0.03, 0.04]},
        "per_relation": {"right": [0.02, 0.01]},
        "per_attribute": {"white": [0.05, -0.01]},
    }
    row = {"prompt_id": "white_right"}
    assert np.allclose(offset_for(row, offsets, "global"), [0.1, -0.2])
    assert np.allclose(offset_for(row, offsets, "per_prompt"), [0.03, 0.04])
    assert np.allclose(offset_for(row, offsets, "per_relation"), [0.02, 0.01])
    assert np.allclose(offset_for(row, offsets, "per_attribute"), [0.05, -0.01])


def test_residual_head_shape_and_normalization() -> None:
    features = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [0.2, 0.1, 0.4, 0.3, 0.6, 0.5, 0.8, 0.7],
        ],
        dtype=torch.float32,
    )
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized = normalize_features(features, mean, std)
    assert normalized.shape == features.shape
    head = DestinationResidualHead(feature_dim=8, hidden_dim=16, dropout=0.0)
    output = head(normalized)
    assert output.shape == (2, 2)
    assert torch.isfinite(output).all()


def test_boundary_features_shape() -> None:
    visual = torch.zeros(3, 5)
    language = torch.ones(3, 5)
    base_object = torch.zeros(3, 2)
    base_destination = torch.ones(3, 2)
    features = boundary_features(
        visual_mean=visual,
        language_mean=language,
        base_object=base_object,
        base_destination=base_destination,
        prompt_ids=["red_left", "red_right", "white_right"],
    )
    assert features.shape == (3, 18)
    assert torch.allclose(features[:3, -4:].sum(dim=1), torch.ones(3))


def _gate_config() -> dict:
    return {
        "gate": {
            "grounding_init_denominator": 15,
            "destination_error_threshold_m": 0.03,
            "destination_mean_error_max_m": 0.015,
            "destination_max_error_max_m": 0.03,
            "destination_correct_min": 12,
            "object_in_mask_min": 12,
        }
    }


def _white_rows(error: float = 0.01) -> list[dict]:
    return [
        {
            "shared_init_id": f"init_{index}",
            "prompt_id": "white_right",
            "destination_error_m": error,
            "destination_position_xyz": [0.0, 0.0, 0.4],
            "object_in_mask": True,
        }
        for index in range(15)
    ]


def test_goal18_gate_pass() -> None:
    gate = evaluate_gate(_white_rows(), config=_gate_config())
    assert gate["status"] == "PASS"
    assert gate["passed"] is True
    assert gate["destination_correct_count"] == 15


def test_goal18_gate_fail() -> None:
    gate = evaluate_gate(_white_rows(error=0.04), config=_gate_config())
    assert gate["status"] == "FAIL"
    assert gate["passed"] is False
    assert "destination_correct" in gate["failed"]


def test_execution_gate_uses_correct_destination_subset() -> None:
    config = {
        "gate": {
            "execution_success_min": 4,
            "execution_success_minus_baseline_min": 2,
            "execution_correct_destination_success_min": 3,
            "execution_wrong_destination_max": 3,
        }
    }
    by_arm = {
        "B0": {"episodes": 15, "success": 0, "errors": 0, "nan_count": 0, "object_flew_out": 0},
        "E1-PRED": {
            "episodes": 15,
            "success": 4,
            "errors": 0,
            "nan_count": 0,
            "object_flew_out": 0,
            "wrong_destination": 2,
        },
    }
    e1_rows = [
        {"target_destination_correct": True, "success": index < 3}
        for index in range(10)
    ] + [{"target_destination_correct": False, "success": False} for _ in range(5)]
    gate = evaluate_goal18_gate(
        by_arm,
        arm_rows={"B0": [], "E1-PRED": e1_rows},
        init_count=15,
        goal_config=config,
        grounding_status={"passed": True},
    )
    assert gate["status"] == "PASS"
    assert gate["correct_destination_subset_success"] == 3
