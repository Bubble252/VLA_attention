"""Data contracts shared by teachers, model adapters, and evaluators.

The objects deliberately carry metadata instead of tensors. Runtime tensors stay
inside adapters so that a Qwen, OpenVLA, or pi0.5 implementation cannot silently
assume the token layout of another model.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence


class CoordinateFrame(str, Enum):
    ORIGINAL_IMAGE = "original_image"
    PATCH_GRID = "patch_grid"
    TEACHER_GRID = "teacher_grid"


class AttributionTarget(str, Enum):
    PHRASE_SCORE = "phrase_score"
    ANSWER_LOGPROB = "answer_logprob"
    ACTION_TOKEN_LOGPROB = "action_token_logprob"
    ACTION_OUTPUT = "action_output"
    ACTION_LOSS = "action_loss"
    FUTURE_SCORE = "future_score"


@dataclass(frozen=True)
class SpatialGrid:
    """One map's geometry and token provenance."""

    height: int
    width: int
    frame: CoordinateFrame
    image_height: int
    image_width: int
    token_indices: tuple[int, ...]
    view_id: str = "single"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if min(self.height, self.width, self.image_height, self.image_width) <= 0:
            raise ValueError("SpatialGrid dimensions must be positive")
        if len(self.token_indices) != self.height * self.width:
            raise ValueError("token_indices must match the declared grid area")
        if len(set(self.token_indices)) != len(self.token_indices):
            raise ValueError("token_indices must be unique within one view")


@dataclass(frozen=True)
class SpatialMap:
    """Metadata for a map stored by an adapter or experiment artifact."""

    map_id: str
    grid: SpatialGrid
    source: str
    phrase: str | None
    target: AttributionTarget | None
    normalization: str
    artifact_path: str | None = None
    provenance: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        self.grid.validate()
        if not self.map_id or not self.source:
            raise ValueError("SpatialMap requires map_id and source")
        if self.target is AttributionTarget.PHRASE_SCORE and not self.phrase:
            raise ValueError("phrase-score attribution requires a phrase")


@dataclass(frozen=True)
class CapabilityReport:
    """P1 output; a model cannot enter a stage without this evidence."""

    model_id: str
    model_revision: str
    supports_visual_tokens: bool
    supports_hidden_states: bool
    supported_targets: tuple[AttributionTarget, ...]
    coordinate_notes: Sequence[str]
    limitations: Sequence[str]
