#!/usr/bin/env python3
"""Fit Goal18 train-only destination offsets and a lightweight residual head.

The frozen Goal17 G4-MARGIN PointQueryHead supplies base destination points.
All offsets and residual labels are computed from train-split mask labels only;
validation is used for reporting, never for gradient updates or arm selection.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_goal16_grounding_sidecar import (  # noqa: E402
    _load_point_head,
    _unproject_table_plane,
)
from scripts.evaluate_goal17_grounding import policy_point_to_raw  # noqa: E402
from src.vla_attention.destination_residual import (  # noqa: E402
    DestinationResidualHead,
    boundary_features,
    normalize_features,
)
from train_c1_point_head import (  # noqa: E402
    PointFeatureDataset,
    _move_batch,
    collate_point_batch,
)


PROMPTS = ("red_left", "red_right", "white_left", "white_right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--val-index", type=Path, required=True)
    parser.add_argument("--point-head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def policy_point_to_world(
    point_xy: Iterable[float],
    *,
    image_size: int,
    raw_resolution: int,
    crop_scale: float,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    table_z_m: float,
) -> np.ndarray:
    raw = policy_point_to_raw(
        point_xy,
        image_size=image_size,
        raw_resolution=raw_resolution,
        crop_scale=crop_scale,
    )
    return _unproject_table_plane(
        raw,
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        table_z_m=table_z_m,
    )


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


@torch.no_grad()
def collect_split(
    *,
    index_path: Path,
    point_head: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    limit: int,
    image: dict[str, Any],
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> dict[str, Any]:
    dataset = PointFeatureDataset(index_path, limit=limit)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_point_batch,
    )
    visual_means: list[torch.Tensor] = []
    language_means: list[torch.Tensor] = []
    base_objects: list[torch.Tensor] = []
    base_destinations: list[torch.Tensor] = []
    prompt_ids: list[str] = []
    rows: list[dict[str, Any]] = []
    point_head.eval()
    for batch in loader:
        batch = _move_batch(batch, device)
        points, _ = point_head(
            batch["visual_tokens"],
            batch["visual_mask"],
            batch["language_tokens"],
            batch["language_mask"],
            visual_view_ids=batch["visual_view_ids"],
        )
        visual_means.append(masked_mean(batch["visual_tokens"].float(), batch["visual_mask"]).cpu())
        language_means.append(
            masked_mean(batch["language_tokens"].float(), batch["language_mask"]).cpu()
        )
        base_objects.append(points[:, 0].float().cpu())
        base_destinations.append(points[:, 1].float().cpu())

        metadata = batch["metadata"]
        batch_size_actual = int(points.shape[0])
        for index in range(batch_size_actual):
            row_id = str(metadata["row_id"][index])
            prompt_id = str(metadata["prompt_id"][index])
            image_size = int(metadata["image_size"][index] or image["image_size"])
            label_dest_px = np.asarray(metadata["destination_point_px"][index], dtype=np.float64)
            label_obj_px = np.asarray(metadata["object_point_px"][index], dtype=np.float64)
            pred_dest_norm = points[index, 1].detach().float().cpu().numpy()
            pred_obj_norm = points[index, 0].detach().float().cpu().numpy()
            pred_dest_px = pred_dest_norm * float(image_size - 1)
            pred_obj_px = pred_obj_norm * float(image_size - 1)
            pred_xyz = policy_point_to_world(
                pred_dest_px,
                image_size=image_size,
                raw_resolution=int(image["raw_resolution"]),
                crop_scale=float(image["crop_scale"]),
                intrinsic=intrinsic,
                extrinsic=extrinsic,
                table_z_m=float(image["table_z_m"]),
            )
            true_xyz = policy_point_to_world(
                label_dest_px,
                image_size=image_size,
                raw_resolution=int(image["raw_resolution"]),
                crop_scale=float(image["crop_scale"]),
                intrinsic=intrinsic,
                extrinsic=extrinsic,
                table_z_m=float(image["table_z_m"]),
            )
            residual = true_xyz[:2] - pred_xyz[:2]
            prompt_ids.append(prompt_id)
            rows.append(
                {
                    "row_id": row_id,
                    "image_id": str(metadata["image_id"][index]),
                    "split": str(metadata["split"][index]),
                    "prompt_id": prompt_id,
                    "image_size": image_size,
                    "base_object_norm": [float(value) for value in pred_obj_norm],
                    "base_destination_norm": [float(value) for value in pred_dest_norm],
                    "label_object_norm": [
                        float(value) / float(image_size - 1) for value in label_obj_px
                    ],
                    "label_destination_norm": [
                        float(value) / float(image_size - 1) for value in label_dest_px
                    ],
                    "base_destination_xyz": [float(value) for value in pred_xyz],
                    "true_destination_xyz": [float(value) for value in true_xyz],
                    "residual_xy_m": [float(value) for value in residual],
                }
            )
    return {
        "rows": rows,
        "visual_mean": torch.cat(visual_means, dim=0),
        "language_mean": torch.cat(language_means, dim=0),
        "base_object": torch.cat(base_objects, dim=0),
        "base_destination": torch.cat(base_destinations, dim=0),
        "prompt_ids": prompt_ids,
    }


def mean_xy(rows: list[dict[str, Any]], predicate) -> list[float]:
    values = [row["residual_xy_m"] for row in rows if predicate(row)]
    if not values:
        raise ValueError("empty residual group")
    return np.asarray(values, dtype=np.float64).mean(axis=0).tolist()


def compute_offsets(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_prompt = {
        prompt: mean_xy(rows, lambda row, p=prompt: row["prompt_id"] == p) for prompt in PROMPTS
    }
    spatial = {
        "left": mean_xy(rows, lambda row: row["prompt_id"].endswith("_left")),
        "right": mean_xy(rows, lambda row: row["prompt_id"].endswith("_right")),
    }
    attribute = {
        "red": mean_xy(rows, lambda row: row["prompt_id"].startswith("red_")),
        "white": mean_xy(rows, lambda row: row["prompt_id"].startswith("white_")),
    }
    return {
        "global": mean_xy(rows, lambda row: True),
        "per_prompt": per_prompt,
        "per_relation": spatial,
        "per_attribute": attribute,
        "counts": {
            "all": len(rows),
            **{prompt: sum(row["prompt_id"] == prompt for row in rows) for prompt in PROMPTS},
        },
    }


def offset_for(row: dict[str, Any], offsets: dict[str, Any], key: str) -> np.ndarray:
    if key == "global":
        return np.asarray(offsets["global"], dtype=np.float64)
    if key == "per_prompt":
        return np.asarray(offsets["per_prompt"][row["prompt_id"]], dtype=np.float64)
    if key == "per_relation":
        relation = "left" if row["prompt_id"].endswith("_left") else "right"
        return np.asarray(offsets["per_relation"][relation], dtype=np.float64)
    if key == "per_attribute":
        attribute = "red" if row["prompt_id"].startswith("red_") else "white"
        return np.asarray(offsets["per_attribute"][attribute], dtype=np.float64)
    raise ValueError(f"unknown offset key: {key}")


def metrics(rows: list[dict[str, Any]], offsets: dict[str, Any] | None, key: str) -> dict[str, Any]:
    errors = []
    for row in rows:
        pred = np.asarray(row["base_destination_xyz"], dtype=np.float64)
        true = np.asarray(row["true_destination_xyz"], dtype=np.float64)
        if offsets is not None:
            pred = pred.copy()
            pred[:2] += offset_for(row, offsets, key)
        errors.append(float(np.linalg.norm(pred[:2] - true[:2])))
    values = np.asarray(errors, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean_m": float(values.mean()),
        "median_m": float(np.median(values)),
        "max_m": float(values.max()),
        "within_0_015_m": int((values <= 0.015).sum()),
        "within_0_03_m": int((values <= 0.03).sum()),
    }


def build_features(
    split: dict[str, Any],
    prompt_ids: list[str],
) -> torch.Tensor:
    return boundary_features(
        visual_mean=split["visual_mean"],
        language_mean=split["language_mean"],
        base_object=split["base_object"],
        base_destination=split["base_destination"],
        prompt_ids=prompt_ids,
    )


def train_residual_head(
    *,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    val_features: torch.Tensor,
    val_targets: torch.Tensor,
    config: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    train_mean = train_features.mean(dim=0)
    train_std = train_features.std(dim=0, unbiased=False).clamp_min(1e-6)
    train_norm = normalize_features(train_features, train_mean, train_std)
    val_norm = normalize_features(val_features, train_mean, train_std)
    model = DestinationResidualHead(
        feature_dim=int(train_norm.shape[1]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_norm = train_norm.to(device)
    train_targets = train_targets.to(device)
    val_norm = val_norm.to(device)
    val_targets = val_targets.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    steps = int(config["steps"])
    batch_size = int(config["batch_size"])
    generator = torch.Generator(device="cpu").manual_seed(int(config.get("seed", 18)))
    best = None
    history: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        indices = torch.randint(0, train_norm.shape[0], (batch_size,), generator=generator)
        batch_x = train_norm[indices.to(device)]
        batch_y = train_targets[indices.to(device)]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch_x)
        loss = torch.nn.functional.mse_loss(prediction, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 250 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                train_pred = model(train_norm)
                val_pred = model(val_norm)
                train_mae = torch.nn.functional.l1_loss(train_pred, train_targets).item()
                val_mae = torch.nn.functional.l1_loss(val_pred, val_targets).item()
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "train_mae_cm": float(train_mae),
                "val_mae_cm": float(val_mae),
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            if best is None or val_mae < best["val_mae_cm"]:
                best = record

    checkpoint = {
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "feature_mean": train_mean.detach().cpu(),
        "feature_std": train_std.detach().cpu(),
        "feature_dim": int(train_norm.shape[1]),
        "config": config,
        "history": history,
        "best": best,
    }
    checkpoint_path = output / "residual_head.pt"
    torch.save(checkpoint, checkpoint_path)
    return {
        "checkpoint": str(checkpoint_path),
        "feature_dim": int(train_norm.shape[1]),
        "history": history,
        "best": best,
        "train_mae_cm": history[-1]["train_mae_cm"] if history else None,
        "val_mae_cm": history[-1]["val_mae_cm"] if history else None,
    }


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = load_json(args.config)
    if config.get("status") != "frozen_before_measurement":
        raise ValueError("Goal18 config must be frozen before measurement")
    image = config["image"]
    camera = load_json(Path(str(config["camera_json"])))["camera"]
    intrinsic = np.asarray(camera["intrinsic_K"], dtype=np.float64)
    extrinsic = np.asarray(camera["camera_env"], dtype=np.float64) if "camera_env" in camera else np.asarray(camera["camera_to_env"], dtype=np.float64)
    table_z_m = float(camera.get("table_z_m", image["table_z_m"]))
    device = torch.device(args.device)
    point_head = _load_point_head(args.point_head, device)

    train = collect_split(
        index_path=args.train_index,
        point_head=point_head,
        device=device,
        batch_size=args.batch_size,
        limit=args.limit_train,
        image=image,
        intrinsic=intrinsic,
        extrinsic=extrinsic,
    )
    val = collect_split(
        index_path=args.val_index,
        point_head=point_head,
        device=device,
        batch_size=args.batch_size,
        limit=args.limit_val,
        image=image,
        intrinsic=intrinsic,
        extrinsic=extrinsic,
    )
    offsets = compute_offsets(train["rows"])

    train_target = torch.tensor(
        [row["residual_xy_m"] for row in train["rows"]], dtype=torch.float32
    ) * float(config["residual_head"]["target_scale_cm"])
    val_target = torch.tensor(
        [row["residual_xy_m"] for row in val["rows"]], dtype=torch.float32
    ) * float(config["residual_head"]["target_scale_cm"])
    train_features = build_features(train, train["prompt_ids"])
    val_features = build_features(val, val["prompt_ids"])

    args.output.mkdir(parents=True)
    residual = train_residual_head(
        train_features=train_features,
        train_targets=train_target,
        val_features=val_features,
        val_targets=val_target,
        config=config["residual_head"],
        output=args.output,
    )

    calibration_payload = {
        "schema": "goal18-destination-calibration-v1",
        "status": "complete",
        "config": str(args.config),
        "train_index": str(args.train_index),
        "val_index": str(args.val_index),
        "point_head": str(args.point_head),
        "offsets": offsets,
        "metrics": {
            "train_base": metrics(train["rows"], None, "none"),
            "train_global": metrics(train["rows"], offsets, "global"),
            "train_prompt": metrics(train["rows"], offsets, "per_prompt"),
            "train_relation": metrics(train["rows"], offsets, "per_relation"),
            "train_attribute": metrics(train["rows"], offsets, "per_attribute"),
            "val_base": metrics(val["rows"], None, "none"),
            "val_global": metrics(val["rows"], offsets, "global"),
            "val_prompt": metrics(val["rows"], offsets, "per_prompt"),
            "val_relation": metrics(val["rows"], offsets, "per_relation"),
            "val_attribute": metrics(val["rows"], offsets, "per_attribute"),
        },
        "residual_head": residual,
        "notes": [
            "Offsets are fitted only on train split mask labels.",
            "Validation metrics are reported, never used for gradient updates or arm selection.",
            "The C1 primary arm is per-prompt; global and relation/attribute are diagnostics.",
            "No eval init, validation frame, simulator success, or future frame is used for training.",
        ],
    }
    (args.output / "calibration.json").write_text(
        json.dumps(calibration_payload, indent=2, ensure_ascii=False) + "\n"
    )
    with (args.output / "train_base_predictions.jsonl").open("w") as handle:
        for row in train["rows"]:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (args.output / "val_base_predictions.jsonl").open("w") as handle:
        for row in val["rows"]:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(calibration_payload, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
