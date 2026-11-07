from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.train_p1_lateral_distill import evaluate  # noqa: E402
from src.vla_attention.p1_student import P1Student  # noqa: E402


def make_data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    count = 6
    return {
        "z_source": rng.normal(size=(count, 16)).astype(np.float32),
        "z_target": rng.normal(size=(count, 16)).astype(np.float32),
        "z_global": rng.normal(size=(count, 16)).astype(np.float32),
        "proprio": rng.normal(size=(count, 8)).astype(np.float32),
        "base_action": rng.normal(size=(count, 8, 7)).astype(np.float32),
        "geometry": rng.normal(size=(count, 6)).astype(np.float32),
        "delta": rng.normal(size=(count, 2)).astype(np.float32),
        "release_target": np.zeros(count, dtype=np.float32),
    }


def make_model() -> P1Student:
    torch.manual_seed(7)
    return P1Student(
        feature_dim=16,
        feature_bottleneck_dim=8,
        hidden_dim=16,
        geometry_dim=6,
        gate_mode="direct_zero_head",
    )


def test_evaluate_returns_correction_probability_and_bounded_xy() -> None:
    model = make_model()
    data = make_data()
    prediction = evaluate(model, data, arm="R2-ROUTE", device="cpu")

    assert prediction["delta"].shape == (len(data["delta"]), 2)
    assert prediction["release_prob"].shape == (len(data["delta"]),)
    assert prediction["correction_prob"].shape == (len(data["delta"]),)
    assert np.isfinite(prediction["delta"]).all()
    assert np.isfinite(prediction["release_prob"]).all()
    assert np.isfinite(prediction["correction_prob"]).all()
    assert np.abs(prediction["delta"]).max() <= 0.1 + 1e-6


def test_evaluate_applies_explicit_correction_threshold() -> None:
    model = make_model()
    data = make_data()
    with torch.no_grad():
        for parameter in model.trunk.parameters():
            parameter.zero_()
        model.xy_head.bias.fill_(1.0)

    prediction = evaluate(
        model,
        data,
        arm="R1-CAP",
        device="cpu",
        correction_threshold=0.75,
    )

    assert np.allclose(prediction["delta"], 0.0)
    assert np.allclose(prediction["correction_prob"], 0.5)
