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

    def validate(self) -> None:
        if not self.model_id or not self.model_revision:
            raise ValueError("CapabilityReport requires model_id and model_revision")
        if not self.supports_visual_tokens:
            raise ValueError("P1 cannot pass a model without visual-token provenance")
        if not self.supported_targets:
            raise ValueError("CapabilityReport must declare at least one target")


@dataclass(frozen=True)
class AuditSample:
    """Immutable input identity for one P1 attribution measurement."""

    sample_id: str
    image_sha256: str
    image_height: int
    image_width: int
    instruction: str
    phrase: str | None = None
    action_schema: str | None = None

    def validate(self) -> None:
        if not self.sample_id or len(self.image_sha256) != 64:
            raise ValueError("AuditSample requires a sample ID and SHA256 image digest")
        if min(self.image_height, self.image_width) <= 0 or not self.instruction:
            raise ValueError("AuditSample requires image dimensions and instruction")


@dataclass(frozen=True)
class AuditMeasurement:
    """One result row. Arrays remain external artifacts, never Git payloads."""

    sample_id: str
    model_id: str
    target: AttributionTarget
    map_path: str
    grid: SpatialGrid
    seed: int
    scalar_definition: str
    gradient_finite: bool
    repeatability: float | None
    action_shape: Sequence[int] | None = None
    runtime_metadata: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        self.grid.validate()
        if not self.sample_id or not self.model_id or not self.map_path:
            raise ValueError("AuditMeasurement requires sample, model, and map artifact")
        if not self.scalar_definition:
            raise ValueError("AuditMeasurement requires a scalar target definition")
        if self.repeatability is not None and not 0 <= self.repeatability <= 1:
            raise ValueError("repeatability must be in [0, 1]")
        is_action = self.target in {
            AttributionTarget.ACTION_OUTPUT,
            AttributionTarget.ACTION_LOSS,
            AttributionTarget.ACTION_TOKEN_LOGPROB,
        }
        if is_action and not self.action_shape:
            raise ValueError("action attribution requires a recorded action_shape")
