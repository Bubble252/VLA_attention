"""Losses shared by controlled VLM V0--V4 experiments."""
from __future__ import annotations


def normalized_spatial_distribution(values, *, epsilon: float = 1e-8):
    """Convert non-negative [batch, tokens] attribution/map mass to probabilities."""
    import torch
    if values.ndim != 2:
        raise ValueError("spatial values must be [batch, tokens]")
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("spatial values must be finite and non-negative")
    return (values + epsilon) / (values.sum(dim=-1, keepdim=True) + epsilon * values.shape[-1])


def semantic_map_kl(student_mass, teacher_mass, *, epsilon: float = 1e-8, teacher_temperature: float = 1.0):
    """KL(teacher || student) on a common image-coordinate token grid.

    ``teacher_temperature < 1`` sharpens spatial peaks and values above one
    soften them, after mass normalization.  The student is left unchanged so
    the parameter isolates conditioning of the fixed SD teacher map.
    """
    import torch
    if teacher_temperature <= 0:
        raise ValueError("teacher_temperature must be positive")
    student = normalized_spatial_distribution(student_mass, epsilon=epsilon)
    teacher = normalized_spatial_distribution(teacher_mass, epsilon=epsilon)
    if teacher_temperature != 1.0:
        teacher = teacher.pow(1.0 / teacher_temperature)
        teacher = teacher / teacher.sum(dim=-1, keepdim=True)
    return (teacher * (teacher.log() - student.log())).sum(dim=-1).mean()
