#!/usr/bin/env python3
"""Merge existing 3-prompt C1 features with the new white_right features.

Goal17 only needs to rerun Qwen for the missing ``white_right`` cell.  This
script validates that the old index covers the other three prompts for every
image and that the new index covers ``white_right`` exactly once per image,
then writes a single 4-prompt index for the grouped counterfactual trainer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROMPTS = ("red_left", "red_right", "white_left", "white_right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-index", type=Path, required=True)
    parser.add_argument("--white-right-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty index: {path}")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_rows(rows: Iterable[dict[str, Any]], *, label: str) -> list[dict[str, Any]]:
    by_image: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        image_id = str(row["image_id"])
        prompt_id = str(row["prompt_id"])
        if prompt_id not in PROMPTS:
            raise ValueError(f"{label}: unknown prompt_id {prompt_id}")
        if prompt_id in by_image[image_id]:
            raise ValueError(f"{label}: duplicate {prompt_id} for {image_id}")
        feature_path = Path(str(row["feature_path"]))
        if not feature_path.is_file():
            raise FileNotFoundError(feature_path)
        by_image[image_id][prompt_id] = row
    return [
        by_image[image_id][prompt]
        for image_id in sorted(by_image)
        for prompt in PROMPTS
        if prompt in by_image[image_id]
    ]


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base_rows = read_jsonl(args.base_index)
    white_rows = read_jsonl(args.white_right_index)
    base_by_image: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in base_rows:
        base_by_image[str(row["image_id"])][str(row["prompt_id"])] = row
    white_by_image: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in white_rows:
        white_by_image[str(row["image_id"])][str(row["prompt_id"])] = row

    if len(base_by_image) != args.expected_images:
        raise ValueError(
            f"base image count {len(base_by_image)} != {args.expected_images}"
        )
    if set(base_by_image) != set(white_by_image):
        missing = sorted(set(base_by_image) - set(white_by_image))[:5]
        extra = sorted(set(white_by_image) - set(base_by_image))[:5]
        raise ValueError(f"image coverage mismatch; missing={missing} extra={extra}")

    merged: list[dict[str, Any]] = []
    for image_id in sorted(base_by_image):
        base_prompts = set(base_by_image[image_id])
        if base_prompts != set(PROMPTS[:3]):
            raise ValueError(f"{image_id}: base prompts are {sorted(base_prompts)}")
        white_prompts = set(white_by_image[image_id])
        if white_prompts != {"white_right"}:
            raise ValueError(f"{image_id}: white-right index prompts {sorted(white_prompts)}")
        white_row = white_by_image[image_id]["white_right"]
        for prompt in PROMPTS[:3]:
            base_row = base_by_image[image_id][prompt]
            if str(base_row.get("mask_path")) != str(white_row.get("mask_path")):
                raise ValueError(f"{image_id}: mask path differs between prompts")
        merged.extend(
            [base_by_image[image_id][prompt] for prompt in PROMPTS[:3]] + [white_row]
        )

    if len(merged) != args.expected_images * 4:
        raise ValueError(f"merged rows {len(merged)} != {args.expected_images * 4}")
    validate_rows(merged, label="merged")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema": "goal17-merged-feature-index-v1",
        "status": "complete",
        "base_index": str(args.base_index),
        "base_index_sha256": sha256_file(args.base_index),
        "white_right_index": str(args.white_right_index),
        "white_right_index_sha256": sha256_file(args.white_right_index),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "image_count": len(base_by_image),
        "row_count": len(merged),
        "prompt_counts": dict(sorted(Counter(str(row["prompt_id"]) for row in merged).items())),
        "notes": [
            "The base index supplies red_left, red_right and white_left.",
            "The white_right features were extracted from the same policy images and mask paths.",
            "No eval or validation image is added to the train index by this script.",
        ],
    }
    (args.output.parent / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
