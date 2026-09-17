"""DINO-to-Qwen retention feature bridge and loss."""
from __future__ import annotations

from typing import Any

from vla_attention.spatial import bilinear_weights


def resample_teacher_features(features: Any, *, source_height: int, source_width: int, target_height: int, target_width: int) -> Any:
    """Bilinearly resample [batch, source_tokens, dim] to target token grid."""
    import torch
    if features.ndim != 3 or features.shape[1] != source_height * source_width:
        raise ValueError("teacher features must be [batch, source_height*source_width, dim]")
    bridge = bilinear_weights(source_height, source_width, target_height, target_width)
    max_width = max(len(row) for row in bridge)
    index_rows, weight_rows = [], []
    for row in bridge:
        indices_row, weights_row = zip(*row)
        index_rows.append((*indices_row, *([indices_row[-1]] * (max_width - len(row)))))
        weight_rows.append((*weights_row, *([0.0] * (max_width - len(row)))))
    indices = torch.tensor(index_rows, device=features.device)
    weights = torch.tensor(weight_rows, dtype=features.dtype, device=features.device)
    gathered = features[:, indices]  # batch, target_tokens, neighbors, dim
    return (gathered * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=2)


def cosine_retention_loss(student: Any, teacher: Any) -> Any:
    """Mean 1-cosine loss on matching patch features."""
    import torch.nn.functional as functional
    if student.shape != teacher.shape:
        raise ValueError(f"retention shapes differ: {student.shape} vs {teacher.shape}")
    return (1.0 - functional.cosine_similarity(student.float(), teacher.float(), dim=-1)).mean()
