"""Qwen2.5-VL geometry helpers used by the native P1 adapter.

Qwen's `image_grid_thw` describes pre-merge visual patches. The language model
receives visual tokens only after the vision encoder's spatial merge. P1 must
record this conversion instead of guessing a square grid from token count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from vla_attention.contracts import CoordinateFrame, SpatialGrid


@dataclass(frozen=True)
class QwenImageGrid:
    temporal: int
    patch_height: int
    patch_width: int
    spatial_merge_size: int

    def validate(self) -> None:
        if min(self.temporal, self.patch_height, self.patch_width, self.spatial_merge_size) <= 0:
            raise ValueError("Qwen grid dimensions and merge size must be positive")
        if self.patch_height % self.spatial_merge_size or self.patch_width % self.spatial_merge_size:
            raise ValueError("patch grid must divide exactly by spatial_merge_size")

    @property
    def merged_height(self) -> int:
        self.validate()
        return self.patch_height // self.spatial_merge_size

    @property
    def merged_width(self) -> int:
        self.validate()
        return self.patch_width // self.spatial_merge_size

    @property
    def visual_token_count(self) -> int:
        return self.temporal * self.merged_height * self.merged_width

    def spatial_grid(self, *, image_height: int, image_width: int, image_token_indices: Sequence[int], view_id: str = "image_0") -> SpatialGrid:
        """Return the post-merge spatial grid for one still image."""
        if self.temporal != 1:
            raise ValueError("P1 2-D map requires a single temporal slice")
        if len(image_token_indices) != self.visual_token_count:
            raise ValueError("image_token_indices must match post-merge visual token count")
        return SpatialGrid(
            height=self.merged_height,
            width=self.merged_width,
            frame=CoordinateFrame.PATCH_GRID,
            image_height=image_height,
            image_width=image_width,
            token_indices=tuple(image_token_indices),
            view_id=view_id,
            metadata={
                "architecture": "qwen2.5-vl",
                "grid_thw": f"{self.temporal},{self.patch_height},{self.patch_width}",
                "spatial_merge_size": str(self.spatial_merge_size),
                "token_stage": "post_spatial_merge",
            },
        )
