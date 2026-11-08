#!/usr/bin/env python3
"""Evaluate Goal18 destination offsets and residual correction.

The frozen Goal17 ``grounding_rows.jsonl`` supplies the base destination point.
Goal18 only applies a train-only 2D correction from either the calibration
file (C1) or the lightweight residual head (C2).  Oracle destinations are used
only for scoring and never enter a deployable arm.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_goal16_grounding_sidecar import _project_point  # noqa: E402
from scripts.evaluate_goal17_grounding import (  # noqa: E402
    load_masks,
    point_in_mask,
    policy_crop_box,
)
from src.vla_attention.destination_residual import (  # noqa: E402
    DestinationResidualHead,
    boundary_features,
    normalize_features,
)


PROMPTS = ("red_left", "red_right", "white_left", "white_right")
EXPECTED_INITS = 15
EXPECTED_ROWS = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-rows", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--residual-head", type=Path, default=None)
    parser.add_argument("--eval-feature-index", type=Path, default=None)
    parser.add_argument("--sidecar", type=Path, default=None)
    parser.add_argument("--arm", default="C1-PROMPT")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--raw-resolution", type=int, default=256)
    parser.add_argument("--crop-scale", type=float, default=0.9)
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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


def raw_to_policy_point(
    raw_point: Iterable[float],
    *,
    image_size: int,
    raw_resolution: int,
    crop_scale: float,
) -> tuple[float, float]:
    box = policy_crop_box(image_size, crop_scale)
    crop_width = box[2] - box[0]
    crop_height = box[3] - box[1]
    x_crop = float(raw_point[0]) * float(image_size) / float(raw_resolution)
    y_crop = float(raw_point[1]) * float(image_size) / float(raw_resolution)
    return (
        (x_crop - box[0]) * float(image_size) / float(crop_width),
        (y_crop - box[1]) * float(image_size) / float(crop_height),
    )


def offset_for(
    row: Mapping[str, Any],
    offsets: Mapping[str, Any],
    key: str,
) -> np.ndarray:
    prompt_id = str(row["prompt_id"])
    if key == "global":
        return np.asarray(offsets["global"], dtype=np.float64)
    if key == "per_prompt":
        return np.asarray(offsets["per_prompt"][prompt_id], dtype=np.float64)
    if key == "per_relation":
        relation = "left" if prompt_id.endswith("_left") else "right"
        return np.asarray(offsets["per_relation"][relation], dtype=np.float64)
    if key == "per_attribute":
        attribute = "red" if prompt_id.startswith("red_") else "white"
        return np.asarray(offsets["per_attribute"][attribute], dtype=np.float64)
    raise ValueError(f"unknown offset key: {key}")


def load_eval_feature_means(index_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    features: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row in read_jsonl(index_path):
        row_id = str(row["row_id"])
        if row_id in features:
            raise ValueError(f"duplicate feature row: {row_id}")
        with np.load(Path(str(row["feature_path"]))) as payload:
            visual = np.asarray(payload["visual_tokens"], dtype=np.float32)
            language = np.asarray(payload["language_tokens"], dtype=np.float32)
        if visual.ndim != 2 or language.ndim != 2:
            raise ValueError(f"{row_id}: feature tensors must be 2D")
        features[row_id] = (visual.mean(axis=0), language.mean(axis=0))
    return features


def build_residual_model(
    checkpoint_path: Path,
    *,
    device: Any,
) -> tuple[DestinationResidualHead, np.ndarray, np.ndarray, float]:
    import torch

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model = DestinationResidualHead(
        feature_dim=int(checkpoint["feature_dim"]),
        hidden_dim=int(config.get("hidden_dim", 256)),
        dropout=float(config.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    mean = np.asarray(checkpoint["feature_mean"].detach().cpu(), dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"].detach().cpu(), dtype=np.float32)
    target_scale = float(config.get("target_scale_cm", 100.0))
    return model, mean, std, target_scale


def apply_c2_residual(
    rows: list[dict[str, Any]],
    *,
    feature_index: Path,
    checkpoint_path: Path,
) -> dict[str, np.ndarray]:
    import torch

    features = load_eval_feature_means(feature_index)
    missing = [str(row["row_id"]) for row in rows if str(row["row_id"]) not in features]
    if missing:
        raise ValueError(f"missing eval features: {missing[:5]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std, target_scale = build_residual_model(checkpoint_path, device=device)
    visual = np.stack([features[str(row["row_id"])][0] for row in rows])
    language = np.stack([features[str(row["row_id"])][1] for row in rows])
    base_object = np.asarray(
        [row["object_point_norm"] for row in rows], dtype=np.float32
    )
    base_destination = np.asarray(
        [row["destination_point_norm"] for row in rows], dtype=np.float32
    )
    prompt_ids = [str(row["prompt_id"]) for row in rows]
    x = boundary_features(
        visual_mean=torch.from_numpy(visual),
        language_mean=torch.from_numpy(language),
        base_object=torch.from_numpy(base_object),
        base_destination=torch.from_numpy(base_destination),
        prompt_ids=prompt_ids,
    )
    x = normalize_features(
        x,
        torch.from_numpy(mean),
        torch.from_numpy(std),
    ).to(device)
    with torch.no_grad():
        prediction_cm = model(x).detach().cpu().numpy()
    residual_m = prediction_cm / target_scale
    return {str(row["row_id"]): residual_m[index] for index, row in enumerate(rows)}


def _sidecar_rows(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    return {str(row["row_id"]): row for row in read_jsonl(path)}


def project_for_mask(
    xyz: Iterable[float],
    *,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    image_size: int,
    raw_resolution: int,
    crop_scale: float,
) -> tuple[float, float]:
    raw = _project_point(xyz, intrinsic=intrinsic, extrinsic=extrinsic)
    if mask.shape == (image_size, image_size):
        return raw_to_policy_point(
            raw,
            image_size=image_size,
            raw_resolution=raw_resolution,
            crop_scale=crop_scale,
        )
    return raw


def evaluate_gate(
    rows: list[dict[str, Any]],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    white_rows = [row for row in rows if str(row["prompt_id"]) == "white_right"]
    gate = config["gate"]
    errors = np.asarray([float(row["destination_error_m"]) for row in white_rows])
    checks = {
        "prediction_count": len(white_rows) == int(gate["grounding_init_denominator"]),
        "destination_correct": int((errors <= float(gate["destination_error_threshold_m"])).sum())
        >= int(gate["destination_correct_min"]),
        "destination_mean": float(errors.mean()) <= float(gate["destination_mean_error_max_m"]),
        "destination_max": float(errors.max()) <= float(gate["destination_max_error_max_m"]),
        "object_in_mask": sum(bool(row["object_in_mask"]) for row in white_rows)
        >= int(gate["object_in_mask_min"]),
        "no_nan": all(
            np.isfinite(np.asarray(row["destination_position_xyz"], dtype=np.float64)).all()
            for row in rows
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "passed": all(checks.values()),
        "checks": checks,
        "failed": [key for key, value in checks.items() if not value],
        "count": len(white_rows),
        "destination_correct_count": int(
            (errors <= float(gate["destination_error_threshold_m"])).sum()
        ),
        "object_in_mask_count": sum(bool(row["object_in_mask"]) for row in white_rows),
        "mean_destination_error_m": float(errors.mean()),
        "max_destination_error_m": float(errors.max()),
    }


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = read_json(args.config)
    if config.get("status") != "frozen_before_measurement":
        raise ValueError("Goal18 config must be frozen before measurement")
    if config.get("oracle_fallback_allowed") is not False:
        raise ValueError("oracle fallback must be disabled")
    arm = str(args.arm)
    if arm not in config.get("arms", {}):
        raise ValueError(f"unknown arm: {arm}")

    rows = read_jsonl(args.base_rows)
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} base rows, got {len(rows)}")
    group_by_init(rows)
    calibration = read_json(args.calibration)
    offsets = calibration["offsets"]
    correction: dict[str, np.ndarray] = {str(row["row_id"]): np.zeros(2, dtype=np.float64) for row in rows}
    if arm.startswith("C1-"):
        key = str(config["arms"][arm]["offset_key"])
        correction = {str(row["row_id"]): offset_for(row, offsets, key) for row in rows}
    elif arm == "C2-RESIDUAL":
        if args.residual_head is None or args.eval_feature_index is None:
            raise ValueError("C2-RESIDUAL requires --residual-head and --eval-feature-index")
        correction = apply_c2_residual(
            rows,
            feature_index=args.eval_feature_index,
            checkpoint_path=args.residual_head,
        )
    elif arm != "C0":
        raise ValueError(f"unsupported correction arm: {arm}")

    sidecar = _sidecar_rows(args.sidecar)
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        corrected = np.asarray(row["destination_position_xyz"], dtype=np.float64).copy()
        corrected[:2] += correction[str(row["row_id"])]
        oracle = np.asarray(row["oracle_destination_xyz"], dtype=np.float64)
        destination_error_m = float(np.linalg.norm(corrected[:2] - oracle[:2]))
        output = dict(row)
        output["destination_position_xyz"] = corrected.tolist()
        output["destination_error_m"] = destination_error_m
        output["grounding_miss"] = destination_error_m > float(
            config["gate"]["destination_error_threshold_m"]
        )
        output["correction_xy_m"] = correction[str(row["row_id"])].tolist()
        output["correction_arm"] = arm
        output["object_in_mask"] = bool(row["object_in_mask"])
        output["destination_in_mask"] = row.get("destination_in_mask")

        mask_row = sidecar.get(str(row["row_id"]))
        if args.sidecar is not None and mask_row is not None:
            masks = load_masks(Path(str(mask_row["mask_path"])))
            target = str(row["target_receptacle"])
            if target not in masks:
                raise ValueError(f"{row['row_id']}: missing mask {target}")
            camera = read_json(Path(str(config["camera_json"])))["camera"]
            intrinsic = np.asarray(camera["intrinsic_K"], dtype=np.float64)
            extrinsic = np.asarray(
                camera["camera_env"] if "camera_env" in camera else camera["camera_to_env"],
                dtype=np.float64,
            )
            point = project_for_mask(
                corrected,
                mask=masks[target],
                intrinsic=intrinsic,
                extrinsic=extrinsic,
                image_size=args.image_size,
                raw_resolution=args.raw_resolution,
                crop_scale=args.crop_scale,
            )
            output["destination_mask_point_px"] = [float(value) for value in point]
            output["destination_in_mask"] = bool(point_in_mask(point, masks[target]))
        out_rows.append(output)

    gate = evaluate_gate(out_rows, config=config)
    args.output.mkdir(parents=True)
    white_rows = [row for row in out_rows if str(row["prompt_id"]) == "white_right"]
    with (args.output / "predicted_points.jsonl").open("w") as handle:
        for row in white_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (args.output / "grounding_rows.jsonl").open("w") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    per_init = []
    by_init: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in out_rows:
        by_init[str(row["shared_init_id"])][str(row["prompt_id"])] = row
    for init_id in sorted(by_init):
        white = by_init[init_id]["white_right"]
        per_init.append(
            {
                "shared_init_id": init_id,
                "split": white.get("split"),
                "destination_error_m": white["destination_error_m"],
                "object_in_mask": white["object_in_mask"],
                "destination_in_mask": white.get("destination_in_mask"),
            }
        )

    summary = {
        "schema": "goal18-grounding-eval-v1",
        "status": gate["status"],
        "passed": gate["passed"],
        "arm": arm,
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "base_rows": str(args.base_rows),
        "base_rows_sha256": sha256_file(args.base_rows),
        "calibration": str(args.calibration),
        "calibration_sha256": sha256_file(args.calibration),
        "residual_head": None if args.residual_head is None else str(args.residual_head),
        "residual_head_sha256": None
        if args.residual_head is None
        else sha256_file(args.residual_head),
        "eval_feature_index": None
        if args.eval_feature_index is None
        else str(args.eval_feature_index),
        "eval_feature_index_sha256": None
        if args.eval_feature_index is None
        else sha256_file(args.eval_feature_index),
        "sidecar": None if args.sidecar is None else str(args.sidecar),
        "sidecar_sha256": None if args.sidecar is None else sha256_file(args.sidecar),
        "predicted_sidecar": str(args.output / "predicted_points.jsonl"),
        "predicted_sidecar_sha256": sha256_file(args.output / "predicted_points.jsonl"),
        "grounding_rows": str(args.output / "grounding_rows.jsonl"),
        "grounding_rows_sha256": sha256_file(args.output / "grounding_rows.jsonl"),
        "gate": gate,
        "per_init": per_init,
        "notes": [
            "Oracle destinations are used only for scoring, never as model input.",
            "C1 offsets and C2 residual weights are fitted on train-split mask labels only.",
            "The frozen Goal17 point head is not modified.",
        ],
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", flush=True)
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
