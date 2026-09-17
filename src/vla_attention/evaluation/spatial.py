"""Grid-to-original-image grounding metrics shared by teachers and students."""
from __future__ import annotations

from typing import Iterable


def _inside(x: float, y: float, boxes: Iterable[Iterable[float]]) -> bool:
    return any(left <= x <= right and top <= y <= bottom for left, top, right, bottom in boxes)


def pointing_correct(grid, boxes, *, image_width: int, image_height: int) -> bool:
    """Whether the maximum map token center falls in at least one GT box."""
    import numpy as np
    if grid.ndim != 2 or min(image_width, image_height) <= 0:
        raise ValueError("grid must be 2-D and image dimensions positive")
    y, x = np.unravel_index(int(np.argmax(grid)), grid.shape)
    center_x, center_y = (x + 0.5) * image_width / grid.shape[1], (y + 0.5) * image_height / grid.shape[0]
    return _inside(center_x, center_y, boxes)


def mass_in_boxes(grid, boxes, *, image_width: int, image_height: int) -> float:
    """Normalized attribution mass whose token centers lie in the union of boxes."""
    import numpy as np
    if grid.ndim != 2 or (grid < 0).any() or not np.isfinite(grid).all():
        raise ValueError("grid must be finite non-negative 2-D mass")
    total = float(grid.sum())
    if total <= 0:
        raise ValueError("grid must have positive mass")
    selected = 0.0
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if _inside((x + 0.5) * image_width / grid.shape[1], (y + 0.5) * image_height / grid.shape[0], boxes):
                selected += float(grid[y, x])
    return selected / total
