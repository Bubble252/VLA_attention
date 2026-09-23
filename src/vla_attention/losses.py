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


def _temperature_teacher(teacher_mass, *, epsilon: float, teacher_temperature: float):
    import torch
    if teacher_temperature <= 0:
        raise ValueError("teacher_temperature must be positive")
    teacher = normalized_spatial_distribution(teacher_mass, epsilon=epsilon)
    if teacher_temperature != 1.0:
        teacher = teacher.pow(1.0 / teacher_temperature)
        teacher = teacher / teacher.sum(dim=-1, keepdim=True)
    return teacher


def semantic_map_js(student_mass, teacher_mass, *, epsilon: float = 1e-8, teacher_temperature: float = 1.0):
    """Symmetric Jensen-Shannon distance on a common spatial grid."""
    student = normalized_spatial_distribution(student_mass, epsilon=epsilon)
    teacher = _temperature_teacher(teacher_mass, epsilon=epsilon, teacher_temperature=teacher_temperature)
    midpoint = 0.5 * (student + teacher)
    kl_student = (student * (student.log() - midpoint.log())).sum(dim=-1)
    kl_teacher = (teacher * (teacher.log() - midpoint.log())).sum(dim=-1)
    return (0.5 * (kl_student + kl_teacher)).mean()


def semantic_map_rank(student_mass, teacher_mass, *, epsilon: float = 1e-8,
                      teacher_temperature: float = 1.0, quantile: float = 0.25,
                      margin: float = 0.5):
    """Teacher top-vs-bottom quantile ranking loss.

    The comparison is made in log mass, so the margin is independent of the
    number of visual tokens. The teacher is detached by construction: it is a
    fixed pseudo-label, never an optimization target.
    """
    import torch
    if not 0.0 < quantile < 0.5:
        raise ValueError("quantile must lie in (0, .5)")
    student = normalized_spatial_distribution(student_mass, epsilon=epsilon)
    teacher = _temperature_teacher(teacher_mass, epsilon=epsilon, teacher_temperature=teacher_temperature).detach()
    n = student.shape[-1]
    k = max(1, int(round(n * quantile)))
    top = torch.topk(teacher, k=k, dim=-1).indices
    bottom = torch.topk(teacher, k=k, dim=-1, largest=False).indices
    top_student = student.gather(-1, top).mean(dim=-1)
    bottom_student = student.gather(-1, bottom).mean(dim=-1)
    return torch.relu(student.new_tensor(margin) - (top_student + epsilon).log() + (bottom_student + epsilon).log()).mean()


def semantic_map_moment(student_mass, teacher_mass, *, epsilon: float = 1e-8,
                        teacher_temperature: float = 1.0, moment_scale: float = 1.0):
    """Match center and spatial variance of teacher/student attribution mass."""
    import torch
    student = normalized_spatial_distribution(student_mass, epsilon=epsilon)
    teacher = _temperature_teacher(teacher_mass, epsilon=epsilon, teacher_temperature=teacher_temperature).detach()
    n = student.shape[-1]
    side = int(round(n ** 0.5))
    if side * side != n:
        raise ValueError("moment loss requires a square common spatial grid")
    axis = torch.linspace(0.0, 1.0, side, device=student.device, dtype=student.dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    coords = torch.stack((yy.reshape(-1), xx.reshape(-1)), dim=-1)
    mu_s = student @ coords
    mu_t = teacher @ coords
    centered_s = coords.unsqueeze(0) - mu_s.unsqueeze(1)
    centered_t = coords.unsqueeze(0) - mu_t.unsqueeze(1)
    cov_s = torch.einsum("bn,bni,bnj->bij", student, centered_s, centered_s)
    cov_t = torch.einsum("bn,bni,bnj->bij", teacher, centered_t, centered_t)
    return moment_scale * ((mu_s - mu_t).square().mean() + (cov_s - cov_t).square().mean())


def semantic_map_loss(student_mass, teacher_mass, *, mode: str = "kl", epsilon: float = 1e-8,
                      teacher_temperature: float = 1.0, rank_quantile: float = 0.25,
                      rank_margin: float = 0.5, rank_weight: float = 1.0,
                      moment_weight: float = 1.0):
    """Select one controlled semantic-map objective.

    ``kl`` is the historical baseline, ``js`` is symmetric, and the two
    composite modes add exactly one geometry term. Keeping the mode explicit
    makes each experiment reproducible and prevents accidental loss mixing.
    """
    if mode == "kl":
        return semantic_map_kl(student_mass, teacher_mass, epsilon=epsilon, teacher_temperature=teacher_temperature)
    if mode == "js":
        return semantic_map_js(student_mass, teacher_mass, epsilon=epsilon, teacher_temperature=teacher_temperature)
    if mode == "kl_rank":
        return semantic_map_kl(student_mass, teacher_mass, epsilon=epsilon, teacher_temperature=teacher_temperature) + rank_weight * semantic_map_rank(
            student_mass, teacher_mass, epsilon=epsilon, teacher_temperature=teacher_temperature,
            quantile=rank_quantile, margin=rank_margin)
    if mode == "kl_moment":
        return semantic_map_kl(student_mass, teacher_mass, epsilon=epsilon, teacher_temperature=teacher_temperature) + moment_weight * semantic_map_moment(
            student_mass, teacher_mass, epsilon=epsilon, teacher_temperature=teacher_temperature)
    raise ValueError(f"unknown semantic map loss mode: {mode}")


def resample_spatial_distribution(values, *, source_height: int, source_width: int,
                                  target_height: int = 32, target_width: int = 32):
    """Differentiably map flattened mass to a fixed common spatial grid."""
    import torch
    import torch.nn.functional as F
    if values.ndim == 2:
        values = values.unsqueeze(1)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values.transpose(1, 2)
    if values.ndim != 3 or values.shape[1] != 1 or values.shape[2] != source_height * source_width:
        raise ValueError("values must be [B,N] or [B,N,1] with N=source_height*source_width")
    image = values.reshape(values.shape[0], 1, source_height, source_width)
    image = F.interpolate(image, size=(target_height, target_width), mode="bilinear", align_corners=False)
    return image.reshape(values.shape[0], target_height * target_width)
