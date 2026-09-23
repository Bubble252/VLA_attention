import pytest
import torch

from vla_attention.losses import (
    normalized_spatial_distribution,
    resample_spatial_distribution,
    semantic_map_js,
    semantic_map_loss,
    semantic_map_moment,
    semantic_map_rank,
)


def test_common_resolution_is_differentiable_and_normalizable():
    x = torch.rand(2, 16, requires_grad=True)
    y = resample_spatial_distribution(x, source_height=4, source_width=4, target_height=8, target_width=8)
    assert y.shape == (2, 64)
    normalized_spatial_distribution(y).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("mode", ["kl", "js", "kl_rank", "kl_moment"])
def test_all_semantic_loss_modes_are_finite(mode):
    student = torch.rand(2, 32 * 32, requires_grad=True)
    teacher = torch.rand(2, 32 * 32)
    value = semantic_map_loss(student, teacher, mode=mode)
    assert torch.isfinite(value)
    value.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()


def test_rank_and_moment_are_zero_like_for_identical_maps():
    x = torch.rand(1, 32 * 32)
    assert semantic_map_moment(x, x).item() < 1e-8
    assert semantic_map_rank(x, x).item() >= 0.0
    assert semantic_map_js(x, x).item() < 1e-8
