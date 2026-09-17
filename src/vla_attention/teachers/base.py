"""Interfaces for frozen spatial teachers and retention baselines."""

from abc import ABC, abstractmethod
from typing import Any

from vla_attention.contracts import SpatialMap


class SpatialTeacher(ABC):
    teacher_id: str

    @abstractmethod
    def map_for_phrase(self, image: Any, phrase: str, *, seed: int) -> SpatialMap:
        """Return a language-conditioned map; weights are always frozen."""

    @abstractmethod
    def calibration_metadata(self) -> dict[str, str]:
        """Return frozen model/configuration identifiers used after calibration."""
