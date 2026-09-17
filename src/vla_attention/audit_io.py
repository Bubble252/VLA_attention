"""Stable JSON representation for P1 interface-audit evidence.

This module intentionally has no torch/transformers dependency. A model adapter
may run in its native repository, then emit these small records for validation.
"""

import json
from pathlib import Path
from typing import Any

from .contracts import AttributionTarget, AuditMeasurement, CapabilityReport, CoordinateFrame, SpatialGrid


def _grid(data: dict[str, Any]) -> SpatialGrid:
    return SpatialGrid(
        height=data["height"], width=data["width"], frame=CoordinateFrame(data["frame"]),
        image_height=data["image_height"], image_width=data["image_width"],
        token_indices=tuple(data["token_indices"]), view_id=data.get("view_id", "single"),
        metadata=data.get("metadata", {}),
    )


def read_report(path: Path) -> tuple[CapabilityReport, list[AuditMeasurement]]:
    """Read and validate the P1 report written by one adapter."""
    data = json.loads(path.read_text())
    capability = CapabilityReport(
        model_id=data["capability"]["model_id"],
        model_revision=data["capability"]["model_revision"],
        supports_visual_tokens=data["capability"]["supports_visual_tokens"],
        supports_hidden_states=data["capability"]["supports_hidden_states"],
        supported_targets=tuple(AttributionTarget(t) for t in data["capability"]["supported_targets"]),
        coordinate_notes=data["capability"].get("coordinate_notes", []),
        limitations=data["capability"].get("limitations", []),
    )
    capability.validate()
    measurements = []
    for row in data["measurements"]:
        measurement = AuditMeasurement(
            sample_id=row["sample_id"], model_id=row["model_id"],
            target=AttributionTarget(row["target"]), map_path=row["map_path"],
            grid=_grid(row["grid"]), seed=row["seed"], scalar_definition=row["scalar_definition"],
            gradient_finite=row["gradient_finite"], repeatability=row.get("repeatability"),
            action_shape=row.get("action_shape"), runtime_metadata=row.get("runtime_metadata", {}),
        )
        measurement.validate()
        if measurement.model_id != capability.model_id:
            raise ValueError("measurement model_id differs from capability model_id")
        if measurement.target not in capability.supported_targets:
            raise ValueError("measurement target was not declared in capability")
        measurements.append(measurement)
    return capability, measurements
