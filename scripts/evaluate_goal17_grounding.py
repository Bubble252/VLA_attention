#!/usr/bin/env python3
"""Evaluate Goal17 PointQueryHead grounding on the frozen 15-init set.

The evaluator uses the T0c same-image sidecar (15 inits x 4 prompts).  It never
reads simulator success or future frames as model input.  Oracle object poses
are used only to score destination error and counterfactual margins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_goal16_grounding_sidecar import (  # noqa: E402
    _load_point_head,
    _point_head_predictions,
    _unproject_table_plane,
)

PROMPTS = ("red_left", "red_right", "white_left", "white_right")
EXPECTED_INITS = 15
EXPECTED_ROWS = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--point-head", type=Path, required=True)
    parser.add_argument("--camera-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-point-head-sha256", default=None)
    parser.add_argument("--expected-sidecar-sha256", default=None)
    parser.add_argument("--max-destination-error-m", type=float, default=0.15)
    parser.add_argument("--margin-threshold", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--raw-resolution", type=int, default=256)
    parser.add_argument("--crop-scale", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty jsonl: {path}")
    return rows


def policy_crop_box(image_size: int, crop_scale: float) -> tuple[int, int, int, int]:
    crop = float(np.sqrt(float(crop_scale)))
    offset = (1.0 - crop) / 2.0
    box = (
        offset * image_size,
        offset * image_size,
        (offset + crop) * image_size,
        (offset + crop) * image_size,
    )
    return tuple(int(round(value)) for value in box)


def policy_point_to_raw(
    point_xy: Iterable[float],
    *,
    image_size: int,
    raw_resolution: int,
    crop_scale: float,
) -> tuple[float, float]:
    box = policy_crop_box(image_size, crop_scale)
    crop_width = box[2] - box[0]
    crop_height = box[3] - box[1]
    x_crop = float(point_xy[0]) * crop_width / float(image_size) + box[0]
    y_crop = float(point_xy[1]) * crop_height / float(image_size) + box[1]
    return (
        x_crop * float(raw_resolution) / float(image_size),
        y_crop * float(raw_resolution) / float(image_size),
    )


def load_masks(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {
            key.removeprefix("mask__"): np.asarray(payload[key], dtype=bool)
            for key in payload.files
            if key.startswith("mask__")
        }


def point_in_mask(point_xy: Iterable[float], mask: np.ndarray) -> bool:
    height, width = mask.shape
    x = int(round(float(point_xy[0])))
    y = int(round(float(point_xy[1])))
    if x < 0 or y < 0 or x >= width or y >= height:
        return False
    return bool(mask[y, x])


def point_for_mask(
    policy_point: Iterable[float],
    raw_point: Iterable[float],
    mask: np.ndarray,
    *,
    image_size: int,
) -> tuple[float, float]:
    if mask.shape == (image_size, image_size):
        return float(policy_point[0]), float(policy_point[1])
    return float(raw_point[0]), float(raw_point[1])


def group_by_init(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        init_id = str(row["shared_init_id"])
        prompt_id = str(row["prompt_id"])
        if prompt_id in groups[init_id]:
            raise ValueError(f"duplicate {init_id}/{prompt_id}")
        groups[init_id][prompt_id] = row
    if len(groups) != EXPECTED_INITS:
        raise ValueError(f"expected {EXPECTED_INITS} inits, got {len(groups)}")
    for init_id, prompts in groups.items():
        if set(prompts) != set(PROMPTS):
            raise ValueError(f"{init_id}: expected {PROMPTS}, got {sorted(prompts)}")
    return groups


def pair_margin(
    *,
    point_a: Iterable[float],
    mask_a: np.ndarray,
    mask_a_wrong: np.ndarray,
    point_b: Iterable[float],
    mask_b: np.ndarray,
    mask_b_wrong: np.ndarray,
) -> float:
    a_correct = point_in_mask(point_a, mask_a)
    a_wrong = point_in_mask(point_a, mask_a_wrong)
    b_correct = point_in_mask(point_b, mask_b)
    b_wrong = point_in_mask(point_b, mask_b_wrong)
    return 0.5 * (
        float(a_correct) - float(a_wrong) + float(b_correct) - float(b_wrong)
    )


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    if config.get("status") != "frozen_before_measurement":
        raise ValueError("config is not frozen")
    if config.get("oracle_fallback_allowed") is not False:
        raise ValueError("oracle fallback must be disabled")
    sidecar_sha = sha256_file(args.sidecar)
    point_head_sha = sha256_file(args.point_head)
    if args.expected_sidecar_sha256 and sidecar_sha != args.expected_sidecar_sha256:
        raise ValueError(f"sidecar SHA mismatch: {sidecar_sha}")
    if args.expected_point_head_sha256 and point_head_sha != args.expected_point_head_sha256:
        raise ValueError(f"point head SHA mismatch: {point_head_sha}")

    camera = json.loads(args.camera_json.read_text())["camera"]
    intrinsic = np.asarray(camera["intrinsic_K"], dtype=np.float64)
    extrinsic = np.asarray(camera["camera_to_env"], dtype=np.float64)
    table_z = float(camera["table_z_m"])

    rows = read_jsonl(args.sidecar)
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} sidecar rows, got {len(rows)}")
    groups = group_by_init(rows)

    import torch
    from omegaconf import OmegaConf
    from starVLA.model.modules.vlm.QWen2_5 import _QWen_VL_Interface

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    interface = _QWen_VL_Interface(config=OmegaConf.load(str(config["qwen_point_config"])))
    interface.model.to(device)
    interface.model.eval()
    point_head = _load_point_head(args.point_head, device)

    args.output.mkdir(parents=True)
    out_rows: list[dict[str, Any]] = []
    masks_by_init: dict[str, dict[str, np.ndarray]] = {}
    for init_id in sorted(groups):
        prompt_row = groups[init_id][PROMPTS[0]]
        masks_by_init[init_id] = load_masks(Path(str(prompt_row["mask_path"])))
        image = Image.open(prompt_row["image_path"]).convert("RGB")
        wrist = Image.open(prompt_row["wrist_image_path"]).convert("RGB")
        if image.size != (args.image_size, args.image_size):
            raise ValueError(f"{init_id}: agent image is {image.size}, expected {args.image_size}")
        for prompt_id in PROMPTS:
            row = groups[init_id][prompt_id]
            points, diagnostics = _point_head_predictions(
                interface=interface,
                point_head=point_head,
                agentview=image,
                wrist=wrist,
                instruction=str(row.get("prompt_language", row["language"])),
            )
            scale = float(args.image_size - 1)
            object_px = points[0] * scale
            destination_px = points[1] * scale
            object_raw = policy_point_to_raw(
                object_px,
                image_size=args.image_size,
                raw_resolution=args.raw_resolution,
                crop_scale=args.crop_scale,
            )
            destination_raw = policy_point_to_raw(
                destination_px,
                image_size=args.image_size,
                raw_resolution=args.raw_resolution,
                crop_scale=args.crop_scale,
            )
            predicted_xyz = _unproject_table_plane(
                destination_raw,
                intrinsic=intrinsic,
                extrinsic=extrinsic,
                table_z_m=table_z,
            )
            oracle_xyz = np.asarray(row["objects"][str(row["target_receptacle"])], dtype=np.float64)
            destination_error_m = float(np.linalg.norm(predicted_xyz[:2] - oracle_xyz[:2]))
            masks = masks_by_init[init_id]
            object_name = str(row["source_object"])
            receptacle_name = str(row["target_receptacle"])
            if object_name not in masks or receptacle_name not in masks:
                raise ValueError(f"{init_id}: missing mask for {object_name}/{receptacle_name}")
            object_mask_point = point_for_mask(
                object_px, object_raw, masks[object_name], image_size=args.image_size
            )
            destination_mask_point = point_for_mask(
                destination_px,
                destination_raw,
                masks[receptacle_name],
                image_size=args.image_size,
            )
            object_in_mask = point_in_mask(object_mask_point, masks[object_name])
            destination_in_mask = point_in_mask(
                destination_mask_point, masks[receptacle_name]
            )
            out_rows.append(
                {
                    "shared_init_id": init_id,
                    "row_id": row["row_id"],
                    "split": row.get("split"),
                    "prompt_id": prompt_id,
                    "prompt_language": row.get("prompt_language", row["language"]),
                    "target_object": object_name,
                    "target_receptacle": receptacle_name,
                    "object_point_norm": [float(value) for value in points[0]],
                    "destination_point_norm": [float(value) for value in points[1]],
                    "object_point_px": [float(value) for value in object_px],
                    "destination_point_px": [float(value) for value in destination_px],
                    "object_point_raw_px": [float(value) for value in object_raw],
                    "destination_point_raw_px": [float(value) for value in destination_raw],
                    "object_mask_point_px": [float(value) for value in object_mask_point],
                    "destination_mask_point_px": [float(value) for value in destination_mask_point],
                    "destination_position_xyz": predicted_xyz.tolist(),
                    "oracle_destination_xyz": oracle_xyz.tolist(),
                    "destination_error_m": destination_error_m,
                    "object_in_mask": bool(object_in_mask),
                    "destination_in_mask": bool(destination_in_mask),
                    "grounding_miss": bool(destination_error_m > args.max_destination_error_m),
                    **diagnostics,
                }
            )
        print(json.dumps({"init": init_id, "rows": len(out_rows)}), flush=True)

    by_init: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in out_rows:
        by_init[str(row["shared_init_id"])][str(row["prompt_id"])] = row

    destination_margin_correct = 0
    object_margin_correct = 0
    destination_pairs_correct = 0
    object_pairs_correct = 0
    per_init: list[dict[str, Any]] = []
    for init_id in sorted(by_init):
        prompts = by_init[init_id]
        masks = masks_by_init[init_id]
        dest_pair_values: list[float] = []
        for object_prompt in ("red", "white"):
            left = prompts[f"{object_prompt}_left"]
            right = prompts[f"{object_prompt}_right"]
            value = pair_margin(
                point_a=left["destination_mask_point_px"],
                mask_a=masks["plate_1"],
                mask_a_wrong=masks["plate_2"],
                point_b=right["destination_mask_point_px"],
                mask_b=masks["plate_2"],
                mask_b_wrong=masks["plate_1"],
            )
            dest_pair_values.append(value)
            destination_pairs_correct += int(value > args.margin_threshold)
        obj_pair_values: list[float] = []
        for receptacle_prompt in ("left", "right"):
            red = prompts[f"red_{receptacle_prompt}"]
            white = prompts[f"white_{receptacle_prompt}"]
            value = pair_margin(
                point_a=red["object_mask_point_px"],
                mask_a=masks["red_coffee_mug_1"],
                mask_a_wrong=masks["porcelain_mug_1"],
                point_b=white["object_mask_point_px"],
                mask_b=masks["porcelain_mug_1"],
                mask_b_wrong=masks["red_coffee_mug_1"],
            )
            obj_pair_values.append(value)
            object_pairs_correct += int(value > args.margin_threshold)
        dest_ok = all(value > args.margin_threshold for value in dest_pair_values)
        obj_ok = all(value > args.margin_threshold for value in obj_pair_values)
        destination_margin_correct += int(dest_ok)
        object_margin_correct += int(obj_ok)
        white = prompts["white_right"]
        per_init.append(
            {
                "shared_init_id": init_id,
                "split": white.get("split"),
                "white_right_destination_error_m": white["destination_error_m"],
                "white_right_object_in_mask": white["object_in_mask"],
                "white_right_destination_in_mask": white["destination_in_mask"],
                "destination_pair_margins": dest_pair_values,
                "object_pair_margins": obj_pair_values,
                "destination_margin_correct": dest_ok,
                "object_margin_correct": obj_ok,
            }
        )

    white_rows = [row for row in out_rows if row["prompt_id"] == "white_right"]
    destination_correct = sum(
        float(row["destination_error_m"]) <= args.max_destination_error_m
        for row in white_rows
    )
    object_in_mask_correct = sum(bool(row["object_in_mask"]) for row in white_rows)
    gate = config["gate"]
    checks = {
        "prediction_count": len(white_rows) == int(gate["grounding_init_denominator"]),
        "destination_correct": destination_correct >= int(gate["destination_correct_min"]),
        "object_in_mask": object_in_mask_correct >= int(gate["object_in_mask_min"]),
        "destination_margin": destination_margin_correct
        >= int(gate["destination_margin_correct_min"]),
        "object_margin": object_margin_correct >= int(gate["object_margin_correct_min"]),
        "no_nan": all(
            np.isfinite(np.asarray(row["destination_position_xyz"], dtype=np.float64)).all()
            for row in out_rows
        ),
    }
    passed = all(checks.values())

    white_right_path = args.output / "predicted_points.jsonl"
    with white_right_path.open("w") as handle:
        for row in white_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    all_path = args.output / "grounding_rows.jsonl"
    with all_path.open("w") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "schema": "goal17-grounding-eval-v1",
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "sidecar": str(args.sidecar),
        "sidecar_sha256": sidecar_sha,
        "point_head": str(args.point_head),
        "point_head_sha256": point_head_sha,
        "camera_json": str(args.camera_json),
        "camera_json_sha256": sha256_file(args.camera_json),
        "predicted_sidecar": str(white_right_path),
        "predicted_sidecar_sha256": sha256_file(white_right_path),
        "sidecar_sha256": sha256_file(white_right_path),
        "grounding_rows": str(all_path),
        "grounding_rows_sha256": sha256_file(all_path),
        "selected_init_count": len(by_init),
        "white_right_row_count": len(white_rows),
        "destination_correct_count": destination_correct,
        "object_in_mask_count": object_in_mask_correct,
        "destination_margin_correct_init_count": destination_margin_correct,
        "object_margin_correct_init_count": object_margin_correct,
        "destination_margin_correct_pair_count": destination_pairs_correct,
        "object_margin_correct_pair_count": object_pairs_correct,
        "max_destination_error_m": max(
            float(row["destination_error_m"]) for row in white_rows
        ),
        "mean_destination_error_m": float(
            np.mean([float(row["destination_error_m"]) for row in white_rows])
        ),
        "checks": checks,
        "per_init": per_init,
        "notes": [
            "Object and destination masks are used only for scoring, never as model inputs.",
            "Counterfactual margins are same-image mask-membership margins.",
            "A per-init margin is correct only when both relevant prompt pairs are correct.",
            "The Gate uses white_right destination error and object-in-mask.",
        ],
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
