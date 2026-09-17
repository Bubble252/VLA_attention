"""Base interface for model-specific adapters.

Adapters may import a cloned external project internally, but no experiment may
reach into references/repos directly.
"""

from abc import ABC, abstractmethod
from typing import Any

from vla_attention.contracts import AttributionTarget, CapabilityReport, SpatialMap


class ModelAdapter(ABC):
    model_id: str

    @abstractmethod
    def audit(self) -> CapabilityReport:
        """Return P1 evidence without training or changing model parameters."""

    @abstractmethod
    def extract_spatial_map(
        self,
        batch: Any,
        target: AttributionTarget,
        *,
        phrase: str | None = None,
    ) -> SpatialMap:
        """Create one auditable map with original grid and target metadata."""
