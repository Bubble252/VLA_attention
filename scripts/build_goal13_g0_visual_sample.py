#!/usr/bin/env python3
"""Build a 100-frame Goal13 G0 visual spot-check contact sheet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image, ImageDraw


TASKS = (
    (
        "red_left",
        "put the red mug on the left plate",
        "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate_demo.hdf5",
        34,
    ),
    (
        "red_right",
        "put the red mug on the right plate",
        "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate_demo.hdf5",
        33,
    ),
    (
        "white_left",
        "put the white mug on the left plate",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_demo.hdf5",
        33,
    ),
)


def sample_task(hdf5_path: Path, task_id: str, language: str, count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with h5py.File(hdf5_path, "r") as handle:
        demos = sorted(handle["data"].keys(), key=lambda name: int(name.split("_")[1]))
        demo_indices = np.linspace(0, len(demos) - 1, 17).round().astype(int).tolist()
        for demo_index in demo_indices:
            demo_name = demos[demo_index]
            frames = handle["data"][demo_name]["obs"]["agentview_rgb"]
            for frame_index in (0, len(frames) - 1):
                if len(rows) >= count:
                    return rows
                rows.append(
                    {
                        "task": task_id,
                        "language": language,
                        "demo": demo_name,
                        "frame": int(frame_index),
                        "steps": int(len(frames)),
                        "source_hdf5": str(hdf5_path),
                        "image": Image.fromarray(frames[frame_index]).convert("RGB"),
                    }
                )
    return rows


def make_contact_sheet(rows: list[dict[str, Any]], output_png: Path) -> None:
    cell_width, cell_height = 160, 176
    columns = 10
    grid_rows = (len(rows) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, grid_rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        column = index % columns
        line = index // columns
        image = row["image"].resize((128, 128), Image.Resampling.NEAREST)
        x = column * cell_width + 16
        y = line * cell_height + 4
        sheet.paste(image, (x, y))
        label = f'{index:03d} {row["task"]}\n{row["demo"]} f{row["frame"]}'
        draw.multiline_text((column * cell_width + 2, y + 132), label, fill="black", spacing=1)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_png)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5-root", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for task_id, language, filename, count in TASKS:
        rows.extend(sample_task(args.hdf5_root / filename, task_id, language, count))
    if len(rows) != 100:
        raise SystemExit(f"expected 100 sampled frames, got {len(rows)}")
    make_contact_sheet(rows, args.output_png)
    manifest_rows = [{key: value for key, value in row.items() if key != "image"} for row in rows]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "schema": "goal13-g0-visual-sample-v1",
                "count": len(manifest_rows),
                "rows": manifest_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "count": len(manifest_rows),
                "png": str(args.output_png),
                "manifest": str(args.output_json),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
