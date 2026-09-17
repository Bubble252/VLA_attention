"""Metrics and protocol checks live here; no model-specific imports."""

from .spatial import mass_in_boxes, pointing_correct

__all__ = ["mass_in_boxes", "pointing_correct"]
