#!/usr/bin/env python3
"""Build Goal17's 2x2 same-image counterfactual point manifests.

The existing C1 training manifests contain three prompts per image:
``red_left``, ``red_right`` and ``white_left``.  Goal17 adds the missing
``white_right`` cell using the already-rendered image and simulator masks.  No
simulator success, future frame, eval image, or eval init is used here.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SOURCE_PROMPTS = ("red_left", "red_right", "white_left")
ALL_PROMPTS = ("red_left", "red_right", "white_left", "white_right")
PROMPT_LANGUAGE = {
    "red_left": "put the red mug on the left plate",
    "red_right": "put the red mug on the right plate",
    "white_left": "put the white mug on the left plate",
    "white_right": "put the white mug on the right plate",
}
PROMPT_SOURCE = {
    "red_left": "red_coffee_mug_1",
    "red_right": "red_coffee_mug_1",
    "white_left": "porcelain_mug_1",
    "white_right": "porcelain_mug_1",
}
PROMPT_DESTINATION = {
    "red_left": "plate_1",
    "red_right": "plate_2",
    "white_left": "plate_1",
    "white_right": "plate_2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-train-images", type=int, default=764)
    parser.add_argument("--expected-val-images", type=int, default=118)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def point_by_object(row: dict[str, Any], object_name: str) -> dict[str, Any]:
    for point in row.get("visible_object_points") or []:
        if str(point.get("object")) == object_name:
            return dict(point)
    raise ValueError(
        f"{row.get('row_id')}: no visible point for {object_name}; "
        f"available={sorted(str(p.get('object')) for p in row.get('visible_object_points') or [])}"
    )


def group_by_image(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        image_id = str(row["image_id"])
        prompt_id = str(row["prompt_id"])
        if prompt_id in groups[image_id]:
            raise ValueError(f"duplicate {prompt_id} for image {image_id}")
        groups[image_id][prompt_id] = row

    for image_id, prompts in groups.items():
        if set(prompts) != set(SOURCE_PROMPTS):
            raise ValueError(f"{image_id}: expected {SOURCE_PROMPTS}, got {sorted(prompts)}")
        base = prompts[SOURCE_PROMPTS[0]]
        for key in ("image_path", "wrist_image_path", "mask_path", "split"):
            values = {str(prompts[prompt].get(key)) for prompt in SOURCE_PROMPTS}
            if len(values) != 1:
                raise ValueError(f"{image_id}: prompt rows disagree on {key}: {sorted(values)}")
        for prompt in SOURCE_PROMPTS:
            row = prompts[prompt]
            if str(row.get("source_object")) != PROMPT_SOURCE[prompt]:
                raise ValueError(f"{row.get('row_id')}: unexpected source_object")
            if str(row.get("target_receptacle")) != PROMPT_DESTINATION[prompt]:
                raise ValueError(f"{row.get('row_id')}: unexpected target_receptacle")
            if str(row.get("language")) != PROMPT_LANGUAGE[prompt]:
                raise ValueError(f"{row.get('row_id')}: unexpected language")
        if not Path(str(base["image_path"])).is_file():
            raise FileNotFoundError(base["image_path"])
        if not Path(str(base["wrist_image_path"])).is_file():
            raise FileNotFoundError(base["wrist_image_path"])
        if not Path(str(base["mask_path"])).is_file():
            raise FileNotFoundError(base["mask_path"])
    return groups


def make_white_right(base: dict[str, Any]) -> dict[str, Any]:
    """Create the missing 2x2 cell from the existing same-image masks."""
    source_object = PROMPT_SOURCE["white_right"]
    destination_name = PROMPT_DESTINATION["white_right"]
    object_point = point_by_object(base, source_object)
    destination_point = point_by_object(base, destination_name)
    image_id = str(base["image_id"])
    row = dict(base)
    row.update(
        {
            "row_id": f"{image_id}__white_right",
            "image_id": image_id,
            "counterfactual_group_id": image_id,
            "prompt_id": "white_right",
            "language": PROMPT_LANGUAGE["white_right"],
            "prompt_language": PROMPT_LANGUAGE["white_right"],
            "source_object": source_object,
            "target_object": destination_name,
            "target_receptacle": destination_name,
            "source_phrase": "white mug",
            "target_phrase": "right plate",
            "phrases": {"relation": "on", "source": "white mug", "target": "right plate"},
            "object_point": object_point,
            "destination_point": destination_point,
            "point_label_source": "goal17_same_image_instance_masks",
            "point_label_usage": "training_label_only",
            "destination_point_kind": "visible_receptacle_surface",
            "deployment_input": False,
        }
    )
    return row


def build_split(
    rows: list[dict[str, Any]],
    *,
    split_name: str,
    expected_images: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups = group_by_image(rows)
    if len(groups) != expected_images:
        raise ValueError(
            f"{split_name}: expected {expected_images} images, got {len(groups)}"
        )
    output: list[dict[str, Any]] = []
    for image_id in sorted(groups):
        prompts = groups[image_id]
        base = prompts[SOURCE_PROMPTS[0]]
        for prompt_id in SOURCE_PROMPTS:
            output.append(dict(prompts[prompt_id]))
        output.append(make_white_right(base))

    if len(output) != expected_images * 4:
        raise ValueError(f"{split_name}: expected {expected_images * 4} rows, got {len(output)}")
    if len({row['row_id'] for row in output}) != len(output):
        raise ValueError(f"{split_name}: duplicate row_id")
    if {row["prompt_id"] for row in output} != set(ALL_PROMPTS):
        raise ValueError(f"{split_name}: prompt coverage is incomplete")
    for image_id in sorted(groups):
        image_rows = [row for row in output if str(row["image_id"]) == image_id]
        if len(image_rows) != 4 or {row["prompt_id"] for row in image_rows} != set(ALL_PROMPTS):
            raise ValueError(f"{image_id}: incomplete 2x2 group")

    summary = {
        "split": split_name,
        "source_rows": len(rows),
        "image_count": len(groups),
        "output_rows": len(output),
        "prompt_counts": dict(sorted(Counter(row["prompt_id"] for row in output).items())),
        "source_object_counts": dict(
            sorted(Counter(row["source_object"] for row in output).items())
        ),
        "target_receptacle_counts": dict(
            sorted(Counter(row["target_receptacle"] for row in output).items())
        ),
        "same_image_counterfactual": True,
        "white_right_source": "visible porcelain_mug_1 and plate_2 masks in the same image",
    }
    return output, summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_white_right(path: Path, rows: list[dict[str, Any]]) -> int:
    selected = [dict(row) for row in rows if str(row.get("prompt_id")) == "white_right"]
    if not selected:
        raise ValueError(f"no white_right rows for {path}")
    write_jsonl(path, selected)
    return len(selected)


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    train_rows, train_summary = build_split(
        read_jsonl(args.train_manifest),
        split_name="train",
        expected_images=args.expected_train_images,
    )
    val_rows, val_summary = build_split(
        read_jsonl(args.val_manifest),
        split_name="validation",
        expected_images=args.expected_val_images,
    )
    train_path = args.output_dir / "train_4prompt.jsonl"
    val_path = args.output_dir / "val_4prompt.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    train_white_right = args.output_dir / "train_white_right.jsonl"
    val_white_right = args.output_dir / "val_white_right.jsonl"
    train_white_right_rows = write_white_right(train_white_right, train_rows)
    val_white_right_rows = write_white_right(val_white_right, val_rows)
    summary = {
        "schema": "goal17-counterfactual-manifest-v1",
        "status": "complete",
        "train_manifest": str(train_path),
        "val_manifest": str(val_path),
        "train_white_right_manifest": str(train_white_right),
        "val_white_right_manifest": str(val_white_right),
        "train_white_right_rows": train_white_right_rows,
        "val_white_right_rows": val_white_right_rows,
        "train": train_summary,
        "validation": val_summary,
        "notes": [
            "Only train-split images are used to synthesize white_right.",
            "No eval init, validation image, success, or future frame is used for labels.",
            "Masks are labels; point-head inputs are only frozen Qwen hidden states.",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
