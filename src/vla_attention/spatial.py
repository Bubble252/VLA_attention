"""Explicit spatial resampling for maps/features on different token grids."""

from __future__ import annotations


def bilinear_weights(source_height: int, source_width: int, target_height: int, target_width: int) -> list[list[tuple[int, float]]]:
    """Token-center bilinear bridge from one raster grid to another.

    It operates in normalized image coordinates, so it is valid for DINO's fixed
    16x16 patches and Qwen's image-dependent post-merge grid.  It does not hide
    a crop/pad transform: callers must first ensure both grids refer to the
    same original-image coordinate frame.
    """
    if min(source_height, source_width, target_height, target_width) <= 0:
        raise ValueError("grid dimensions must be positive")
    rows: list[list[tuple[int, float]]] = []
    for target_y in range(target_height):
        y = ((target_y + 0.5) / target_height) * source_height - 0.5
        y0, y1 = max(0, int(y // 1)), min(source_height - 1, int(y // 1) + 1)
        wy1, wy0 = y - int(y // 1), 1.0 - (y - int(y // 1))
        if y0 == y1:
            wy0, wy1 = 1.0, 0.0
        for target_x in range(target_width):
            x = ((target_x + 0.5) / target_width) * source_width - 0.5
            x0, x1 = max(0, int(x // 1)), min(source_width - 1, int(x // 1) + 1)
            wx1, wx0 = x - int(x // 1), 1.0 - (x - int(x // 1))
            if x0 == x1:
                wx0, wx1 = 1.0, 0.0
            weights: dict[int, float] = {}
            for yy, wy in ((y0, wy0), (y1, wy1)):
                for xx, wx in ((x0, wx0), (x1, wx1)):
                    weights[yy * source_width + xx] = weights.get(yy * source_width + xx, 0.0) + wy * wx
            rows.append(sorted((index, weight) for index, weight in weights.items() if weight > 0.0))
    return rows
