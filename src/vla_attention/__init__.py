"""Reusable contracts for VLM/VLA attribution experiments.

Model-specific code belongs in adapters; experiments must communicate through
the contracts defined here rather than importing external repositories directly.
"""

from .contracts import AttributionTarget, CoordinateFrame, SpatialGrid, SpatialMap

__all__ = ["AttributionTarget", "CoordinateFrame", "SpatialGrid", "SpatialMap"]
