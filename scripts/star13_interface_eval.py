#!/usr/bin/env python3
"""Goal13 offline counterfactual evaluation with I-layer point ablations.

The same image, state and expert action chunk are paired with the correct
instruction and a safe wrong-destination or wrong-object instruction.  The
primary signal is ``wrong_mae - correct_mae``: positive means the correct
instruction produces actions closer to the expert chunk.

Adds predicted/none/zero/shuffle point-marker ablations and an oracle
GT-point diagnostic for the I1 interface arm. The oracle path is diagnostic
only and is never accepted as a deployment input.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file

from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES
from starVLA.dataloader.lerobot_datasets import get_vla_dataset, make_LeRobotSingleDataset
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.model.modules.counterfactual_binding import (
    C1_SEEN_LANGUAGES,
    counterfactual_negative_languages,
)


TRAIN_DATASET = "libero_90_gate_lerobot"
VALIDATION_DATASET = "libero_90_gate_validation_lerobot"


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_action_head(model, checkpoint: Path) -> list[str]:
    state = load_file(str(checkpoint))
    module_prefixes = {
        "action_model.": "action_model",
        "subgoal_head.": "subgoal_head",
        "subgoal_projector.": "subgoal_projector",
        "point_tokenizer.": "point_tokenizer",
        "point_evidence_projector.": "point_evidence_projector",
        "region_head.": "region_head",
        "region_evidence_projector.": "region_evidence_projector",
        "evidence_action.": "evidence_action",
    }
    loaded: list[str] = []
    for prefix, module_name in module_prefixes.items():
        module_state = {
            key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)
        }
        if not module_state:
            continue
        module = getattr(model, module_name, None)
        if module is None:
            raise RuntimeError(f"checkpoint contains {prefix} but model has no {module_name}")
        module.load_state_dict(module_state, strict=True)
        loaded.append(module_name)
    if not loaded:
        raise RuntimeError(f"checkpoint has no recognized trainable modules: {checkpoint}")
    return loaded


def sample_at(dataset, episode_index: int, state_index: int) -> dict:
    raw = dataset.get_step_data(episode_index, state_index)
    return dataset._pack_sample(dataset.transforms(raw))


def binary_iou(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    predicted = predicted.astype(np.float32) > 0.5
    target = target.astype(np.float32) > 0.5
    intersection = np.logical_and(predicted, target).sum(axis=(-2, -1))
    union = np.logical_or(predicted, target).sum(axis=(-2, -1))
    return (intersection + 1.0) / (union + 1.0)


def load_region_targets(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(str(row["mask_path"])) as payload:
        object_mask = np.asarray(payload[f"grid__{row['source_object']}"], dtype=np.float32)
        destination_mask = np.asarray(
            payload[f"grid__{row['target_object']}"], dtype=np.float32
        )
    return object_mask, destination_mask


def load_point_masks(row: dict[str, Any]) -> np.ndarray:
    """Load the source/destination masks used only for point-in-mask checks."""
    with np.load(str(row["mask_path"])) as payload:
        source = np.asarray(payload[f"mask__{row['source_object']}"], dtype=bool)
        destination = np.asarray(payload[f"mask__{row['target_receptacle']}"], dtype=bool)
    return np.stack([source, destination], axis=0)


def oracle_points_from_row(row: dict[str, Any]) -> np.ndarray:
    """Return the two GT points in the same normalized convention as P1."""
    size = int(row.get("policy_image_size", row.get("image_size", 224)))
    if size < 2:
        raise ValueError("image size must be >= 2")
    points = []
    for key in ("object_point", "destination_point"):
        point = row[key]["point_xy"]
        points.append([float(point[0]) / (size - 1), float(point[1]) / (size - 1)])
    return np.asarray(points, dtype=np.float32)


def point_l1_px(predictions: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, float]:
    targets = np.stack([oracle_points_from_row(row) for row in rows])
    error = np.abs(predictions.astype(np.float32) - targets) * (np.asarray([223.0, 223.0], dtype=np.float32))
    return {
        "all": float(error.mean()),
        "object": float(error[:, 0].mean()),
        "destination": float(error[:, 1].mean()),
    }


def point_in_mask_fraction(
    predictions: np.ndarray,
    rows: list[dict[str, Any]],
) -> float:
    hits: list[bool] = []
    for prediction, row in zip(predictions, rows):
        masks = load_point_masks(row)
        for query in range(2):
            x = int(round(float(prediction[query, 0]) * 223.0))
            y = int(round(float(prediction[query, 1]) * 223.0))
            x = max(0, min(223, x))
            y = max(0, min(223, y))
            hits.append(bool(masks[query, y, x]))
    return float(np.mean(hits))


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read either a JSON array/object or a JSONL row manifest."""
    text = path.read_text()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("rows", "episodes", "records"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(
        f"{path} is a whitelist, not an episode-id mapping; pass the LeRobot "
        "meta/c1_source_episodes.jsonl file instead"
    )


def bootstrap_ci(values: np.ndarray, *, seed: int, samples: int = 2000) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        picks = rng.integers(0, len(values), size=len(values))
        means[index] = float(np.mean(values[picks]))
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def action_mae(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.abs(predicted.astype(np.float32) - target.astype(np.float32)).mean(axis=(1, 2))


def amplitude_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    pred_dim = np.abs(predicted).mean(axis=(0, 1))
    target_dim = np.abs(target).mean(axis=(0, 1))
    ratio = pred_dim / np.maximum(target_dim, 1e-12)
    return {
        "predicted_action_abs_mean": float(np.abs(predicted).mean()),
        "target_action_abs_mean": float(np.abs(target).mean()),
        "action_amplitude_ratio": float(np.abs(predicted).mean() / max(np.abs(target).mean(), 1e-12)),
        "translation_amplitude_ratio": float(ratio[:3].mean()),
        "rotation_amplitude_ratio": float(ratio[3:6].mean()),
        "gripper_amplitude_ratio": float(ratio[6]),
        "per_dimension_amplitude_ratio": [float(value) for value in ratio],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--validation-episodes", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--validation-whitelist", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--point-manifest", type=Path, default=None)
    parser.add_argument("--point-manifest-sha256", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument(
        "--region-ablation",
        choices=("auto", "predicted", "shuffle", "none", "zero"),
        default="auto",
    )
    parser.add_argument(
        "--point-ablation",
        choices=("auto", "predicted", "oracle", "shuffle", "none", "zero"),
        default="auto",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = apply_config_compat(OmegaConf.load(args.config))
    train_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    train_cfg.datasets.vla_data.data_root_dir = args.train_root
    train_cfg.datasets.vla_data.episode_whitelist_path = None
    train_cfg.datasets.vla_data.episode_whitelist_by_dataset = None
    train_mix = str(train_cfg.datasets.vla_data.data_mix)
    if train_mix not in DATASET_NAMED_MIXTURES:
        raise KeyError(f"unknown training data_mix: {train_mix}")
    if len(DATASET_NAMED_MIXTURES[train_mix]) > 1:
        train_dataset = get_vla_dataset(
            data_cfg=train_cfg.datasets.vla_data,
            balance_dataset_weights=bool(
                train_cfg.datasets.vla_data.get("balance_dataset_weights", False)
            ),
            balance_trajectory_weights=bool(
                train_cfg.datasets.vla_data.get("balance_trajectory_weights", False)
            ),
        )
        train_metadata = train_dataset.merged_metadata[train_dataset.datasets[0].tag]
    else:
        train_dataset = make_LeRobotSingleDataset(
            args.train_root,
            TRAIN_DATASET,
            "libero_franka",
            data_cfg=train_cfg.datasets.vla_data,
        )
        train_metadata = train_dataset.metadata

    val_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    val_cfg.datasets.vla_data.data_root_dir = args.validation_root
    val_cfg.datasets.vla_data.episode_whitelist_path = args.validation_whitelist
    val_cfg.datasets.vla_data.episode_whitelist_by_dataset = None
    val_dataset = make_LeRobotSingleDataset(
        args.validation_root,
        VALIDATION_DATASET,
        "libero_franka",
        data_cfg=val_cfg.datasets.vla_data,
    )
    # Normalization must be inherited from the training export, not recomputed
    # on the 15 validation episodes.
    val_dataset.set_transforms_metadata(train_metadata)

    validation_rows = read_rows(args.validation_episodes)
    episode_to_index: dict[str, int] = {}
    for row in validation_rows:
        episode_id = row.get("rlds_episode_id", row.get("episode_id"))
        episode_index = row.get("episode_index", row.get("output_record_index"))
        if episode_id is None or episode_index is None:
            raise ValueError(f"validation episode mapping row is incomplete: {row}")
        episode_to_index[str(episode_id)] = int(episode_index)
    sidecar_rows = read_rows(args.sidecar)
    point_lookup: dict[tuple[str, int, str], dict[str, Any]] = {}
    if args.point_ablation != "none":
        if args.point_manifest is None:
            raise ValueError("point ablation requires --point-manifest")
        if not args.point_manifest.is_file():
            raise FileNotFoundError(args.point_manifest)
        if args.point_manifest_sha256 is not None:
            actual_sha = sha256_file(args.point_manifest)
            if actual_sha != args.point_manifest_sha256:
                raise ValueError(
                    "point manifest SHA mismatch: "
                    f"{actual_sha} != {args.point_manifest_sha256}"
                )
        for point_row in read_rows(args.point_manifest):
            point_key = (
                str(point_row["episode_id"]),
                int(point_row.get("state_index", point_row.get("frame_index"))),
                str(point_row["language"]),
            )
            if point_key in point_lookup:
                raise ValueError(f"duplicate point manifest row: {point_key}")
            if "object_point" not in point_row or "destination_point" not in point_row:
                raise ValueError(f"point manifest row missing points: {point_key}")
            point_lookup[point_key] = point_row
    groups: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in sidecar_rows:
        language = str(row["language"])
        if language not in C1_SEEN_LANGUAGES:
            raise ValueError(f"unexpected C1 language: {language}")
        key = (
            str(row["episode_id"]),
            int(row.get("state_index", row.get("frame_index"))),
        )
        if args.point_ablation != "none":
            point_row = point_lookup.get((key[0], key[1], language))
            if point_row is None:
                raise KeyError(f"point manifest missing row: {key + (language,)}")
            row = {
                **row,
                "object_point": point_row["object_point"],
                "destination_point": point_row["destination_point"],
                "policy_image_size": point_row.get(
                    "policy_image_size", row.get("policy_image_size")
                ),
            }
        groups[key][language] = row
    group_items = list(groups.items())
    if args.limit > 0:
        group_items = group_items[: args.limit]
    if not group_items:
        raise ValueError("no validation groups")

    model = build_framework(cfg).to("cuda").eval()
    loaded_modules = load_action_head(model, args.checkpoint)
    if args.residual_scale != 1.0 and hasattr(model, "evidence_action"):
        if model.evidence_action is None:
            raise RuntimeError("residual scale requested but model has no evidence_action")
        with torch.no_grad():
            model.evidence_action.weight.mul_(args.residual_scale)
            model.evidence_action.bias.mul_(args.residual_scale)
    if args.region_ablation == "auto":
        region_ablation = "predicted" if "region_head" in loaded_modules else "zero"
    else:
        region_ablation = args.region_ablation
    point_available = bool(getattr(model, "point_condition_enabled", False)) and bool(
        getattr(model, "point_evidence_projector", None) is not None
    )
    if args.point_ablation == "auto":
        point_ablation = "predicted" if point_available else "none"
    else:
        point_ablation = args.point_ablation
    if point_ablation != "none" and not point_available:
        raise RuntimeError("point ablation requested but model has no point-conditioned projector")

    original_point_evidence = model._point_evidence
    oracle_points_override: torch.Tensor | None = None

    def patched_point_evidence(last_hidden, qwen_inputs, *, point_ablation="predicted"):
        if oracle_points_override is None:
            return original_point_evidence(
                last_hidden, qwen_inputs, point_ablation=point_ablation
            )
        points = oracle_points_override.to(device=last_hidden.device, dtype=torch.float32)
        point_tokens, info = model.point_tokenizer(points)
        return point_tokens.to(dtype=last_hidden.dtype), info["valid"], {
            **info,
            "points": points.detach(),
            "ablation": "oracle",
        }

    model._point_evidence = patched_point_evidence
    for parameter in model.parameters():
        parameter.requires_grad = False

    def predict(
        batch: list[dict[str, Any]],
        seed: int,
        *,
        region_mode: str = region_ablation,
        return_regions: bool = False,
        point_mode: str = point_ablation,
        oracle_points: np.ndarray | None = None,
    ) -> dict[str, Any]:
        nonlocal oracle_points_override
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        oracle_points_override = (
            torch.from_numpy(np.asarray(oracle_points, dtype=np.float32))
            if point_mode == "oracle"
            else None
        )
        try:
            output = model.predict_action(
                batch,
                return_regions=return_regions,
                region_ablation=region_mode,
                point_ablation="predicted" if point_mode == "oracle" else point_mode,
                return_points=point_mode != "none",
            )
        finally:
            oracle_points_override = None
        output = dict(output)
        output["normalized_actions"] = np.asarray(
            output["normalized_actions"], dtype=np.float32
        )
        return output

    samples: list[dict[str, Any]] = []
    for group_key, language_rows in group_items:
        episode_id, state_index = group_key
        if episode_id not in episode_to_index:
            raise KeyError(f"validation episode missing from manifest: {episode_id}")
        sample = sample_at(val_dataset, episode_to_index[episode_id], state_index)
        if str(sample["lang"]) not in C1_SEEN_LANGUAGES:
            raise ValueError(f"validation sample has unexpected language: {sample['lang']}")
        if str(sample["lang"]) not in language_rows:
            raise KeyError(
                f"validation language {sample['lang']!r} missing from sidecar group {group_key}"
            )
        sample["_region_row"] = language_rows[str(sample["lang"])]
        sample["_state_index"] = state_index
        samples.append(sample)

    pair_results: dict[str, dict[str, Any]] = {}
    for kind in ("destination", "object"):
        selected = [
            index
            for index, sample in enumerate(samples)
            if counterfactual_negative_languages(str(sample["lang"]), kind) is not None
        ]
        if not selected:
            continue
        correct_all: list[np.ndarray] = []
        wrong_all: list[np.ndarray] = []
        target_all: list[np.ndarray] = []
        for start in range(0, len(selected), args.batch_size):
            index_chunk = selected[start : start + args.batch_size]
            correct_batch = [samples[index] for index in index_chunk]
            wrong_batch = [
                {
                    **samples[index],
                    "lang": counterfactual_negative_languages(str(samples[index]["lang"]), kind),
                }
                for index in index_chunk
            ]
            oracle = (
                np.stack(
                    [
                        oracle_points_from_row(samples[index]["_region_row"])
                        for index in index_chunk
                    ]
                )
                if point_ablation == "oracle"
                else None
            )
            correct_all.extend(
                predict(
                    correct_batch,
                    args.seed + start,
                    oracle_points=oracle,
                )["normalized_actions"]
            )
            wrong_all.extend(
                predict(
                    wrong_batch,
                    args.seed + start,
                    oracle_points=oracle,
                )["normalized_actions"]
            )
            target_all.extend(
                np.asarray(samples[index]["action"], dtype=np.float32) for index in index_chunk
            )
        correct = np.stack(correct_all)
        wrong = np.stack(wrong_all)
        target = np.stack(target_all)
        correct_mae = action_mae(correct, target)
        wrong_mae = action_mae(wrong, target)
        margin = wrong_mae - correct_mae
        bootstrap = bootstrap_ci(
            margin,
            seed=args.seed + (0 if kind == "destination" else 10000),
            samples=args.bootstrap_samples,
        )
        languages = [str(samples[index]["lang"]) for index in selected]
        by_language: dict[str, Any] = {}
        for language in sorted(set(languages)):
            language_mask = np.asarray([value == language for value in languages])
            language_margin = margin[language_mask]
            by_language[language] = {
                "pair_count": int(language_mask.sum()),
                "correct_mae": float(correct_mae[language_mask].mean()),
                "wrong_mae": float(wrong_mae[language_mask].mean()),
                "margin_mean": float(language_margin.mean()),
                "margin_bootstrap_95ci": bootstrap_ci(
                    language_margin,
                    seed=args.seed + (100 if kind == "destination" else 20000),
                    samples=args.bootstrap_samples,
                ),
            }
        pair_results[kind] = {
            "pair_count": int(len(selected)),
            "correct_mae": float(correct_mae.mean()),
            "wrong_mae": float(wrong_mae.mean()),
            "margin_mean": float(margin.mean()),
            "margin_median": float(np.median(margin)),
            "margin_bootstrap_95ci": bootstrap,
            "margin_positive_fraction": float((margin > 0).mean()),
            "correct_translation_mae": float(np.abs(correct - target)[:, :, :3].mean()),
            "correct_rotation_mae": float(np.abs(correct - target)[:, :, 3:6].mean()),
            "correct_gripper_mae": float(np.abs(correct - target)[:, :, 6].mean()),
            "wrong_translation_mae": float(np.abs(wrong - target)[:, :, :3].mean()),
            "wrong_rotation_mae": float(np.abs(wrong - target)[:, :, 3:6].mean()),
            "wrong_gripper_mae": float(np.abs(wrong - target)[:, :, 6].mean()),
            "action_delta": float(np.abs(wrong - correct).mean()),
            "by_language": by_language,
            **amplitude_metrics(correct, target),
        }

    region_results: dict[str, Any] | None = None
    if "region_head" in loaded_modules:
        predicted_actions: list[np.ndarray] = []
        shuffle_actions: list[np.ndarray] = []
        none_actions: list[np.ndarray] = []
        region_targets: list[np.ndarray] = []
        predicted_masks: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for start in range(0, len(samples), args.batch_size):
            batch = samples[start : start + args.batch_size]
            seed = args.seed + start
            predicted = predict(
                batch, seed, region_mode="predicted", return_regions=True
            )
            shuffled = predict(batch, seed, region_mode="shuffle")
            none = predict(batch, seed, region_mode="none")
            predicted_actions.extend(predicted["normalized_actions"])
            shuffle_actions.extend(shuffled["normalized_actions"])
            none_actions.extend(none["normalized_actions"])
            predicted_masks.extend(np.asarray(predicted["region_probabilities"]))
            for sample in batch:
                object_mask, destination_mask = load_region_targets(sample["_region_row"])
                region_targets.append(np.stack([object_mask, destination_mask], axis=0))
                targets.append(np.asarray(sample["action"], dtype=np.float32))
        predicted_actions_np = np.stack(predicted_actions)
        shuffle_actions_np = np.stack(shuffle_actions)
        none_actions_np = np.stack(none_actions)
        targets_np = np.stack(targets)
        region_targets_np = np.stack(region_targets)
        predicted_masks_np = np.stack(predicted_masks)
        predicted_mae = action_mae(predicted_actions_np, targets_np)
        shuffle_mae = action_mae(shuffle_actions_np, targets_np)
        none_mae = action_mae(none_actions_np, targets_np)
        shuffle_gain = shuffle_mae - predicted_mae
        none_gain = none_mae - predicted_mae
        if len(predicted_masks_np) > 1:
            shuffled_masks = np.roll(predicted_masks_np, 1, axis=0)
        else:
            shuffled_masks = predicted_masks_np
        constant_masks = np.ones_like(region_targets_np[:, 0])
        iou_predicted = np.stack(
            [
                binary_iou(predicted_masks_np[:, query], region_targets_np[:, query])
                for query in range(2)
            ],
            axis=1,
        )
        iou_shuffled = np.stack(
            [
                binary_iou(shuffled_masks[:, query], region_targets_np[:, query])
                for query in range(2)
            ],
            axis=1,
        )
        iou_constant = np.stack(
            [
                binary_iou(constant_masks, region_targets_np[:, query])
                for query in range(2)
            ],
            axis=1,
        )
        region_results = {
            "num_groups": len(samples),
            "action_comparison": {
                "predicted_mae": float(predicted_mae.mean()),
                "shuffle_mae": float(shuffle_mae.mean()),
                "none_mae": float(none_mae.mean()),
                "shuffle_minus_predicted": float(shuffle_gain.mean()),
                "shuffle_minus_predicted_bootstrap_95ci": bootstrap_ci(
                    shuffle_gain,
                    seed=args.seed + 30000,
                    samples=args.bootstrap_samples,
                ),
                "none_minus_predicted": float(none_gain.mean()),
                "none_minus_predicted_bootstrap_95ci": bootstrap_ci(
                    none_gain,
                    seed=args.seed + 40000,
                    samples=args.bootstrap_samples,
                ),
            },
            "iou": {
                "object_predicted": float(iou_predicted[:, 0].mean()),
                "object_shuffle": float(iou_shuffled[:, 0].mean()),
                "object_constant": float(iou_constant[:, 0].mean()),
                "destination_predicted": float(iou_predicted[:, 1].mean()),
                "destination_shuffle": float(iou_shuffled[:, 1].mean()),
                "destination_constant": float(iou_constant[:, 1].mean()),
            },
            "region_ablation": region_ablation,
        }

    point_results: dict[str, Any] | None = None
    if point_available:
        mode_actions: dict[str, list[np.ndarray]] = {
            mode: [] for mode in ("predicted", "oracle", "shuffle", "none")
        }
        mode_points: dict[str, list[np.ndarray]] = {mode: [] for mode in mode_actions}
        point_targets: list[np.ndarray] = []
        point_action_targets: list[np.ndarray] = []
        point_rows: list[dict[str, Any]] = []
        for start in range(0, len(samples), args.batch_size):
            batch = samples[start : start + args.batch_size]
            seed = args.seed + start
            oracle = np.stack(
                [oracle_points_from_row(sample["_region_row"]) for sample in batch]
            )
            for mode in mode_actions:
                output = predict(
                    batch,
                    seed,
                    point_mode=mode,
                    oracle_points=oracle if mode == "oracle" else None,
                )
                mode_actions[mode].extend(output["normalized_actions"])
                if "points" in output:
                    mode_points[mode].extend(np.asarray(output["points"]))
            point_targets.extend(oracle)
            point_action_targets.extend(
                np.asarray(sample["action"], dtype=np.float32) for sample in batch
            )
            point_rows.extend(sample["_region_row"] for sample in batch)

        targets_np = np.stack(point_action_targets)
        mae_by_mode = {
            mode: action_mae(np.stack(actions), targets_np)
            for mode, actions in mode_actions.items()
        }
        amplitude_by_mode = {
            mode: amplitude_metrics(np.stack(actions), targets_np)
            for mode, actions in mode_actions.items()
        }
        point_l1_by_mode = {
            mode: point_l1_px(np.stack(points), point_rows)
            for mode, points in mode_points.items()
            if points
        }
        point_in_mask_by_mode = {
            mode: point_in_mask_fraction(np.stack(points), point_rows)
            for mode, points in mode_points.items()
            if points
        }
        predicted_mae = mae_by_mode["predicted"]
        oracle_mae = mae_by_mode["oracle"]
        shuffle_mae = mae_by_mode["shuffle"]
        none_mae = mae_by_mode["none"]
        point_results = {
            "num_groups": len(samples),
            "action_comparison": {
                "predicted_mae": float(predicted_mae.mean()),
                "oracle_mae": float(oracle_mae.mean()),
                "shuffle_mae": float(shuffle_mae.mean()),
                "none_mae": float(none_mae.mean()),
                "oracle_minus_predicted": float((predicted_mae - oracle_mae).mean()),
                "predicted_minus_none": float((none_mae - predicted_mae).mean()),
                "predicted_minus_shuffle": float((shuffle_mae - predicted_mae).mean()),
                "predicted_minus_shuffle_bootstrap_95ci": bootstrap_ci(
                    shuffle_mae - predicted_mae,
                    seed=args.seed + 30000,
                    samples=args.bootstrap_samples,
                ),
                "predicted_minus_none_bootstrap_95ci": bootstrap_ci(
                    none_mae - predicted_mae,
                    seed=args.seed + 40000,
                    samples=args.bootstrap_samples,
                ),
            },
            "amplitude": amplitude_by_mode,
            "point_l1_px": point_l1_by_mode,
            "point_in_mask": point_in_mask_by_mode,
            "point_ablation": point_ablation,
        }

    destination_passed = (
        "destination" in pair_results
        and pair_results["destination"]["margin_mean"] > 0.0
        and pair_results["destination"]["margin_bootstrap_95ci"][0] > 0.0
    )
    object_passed = (
        "object" in pair_results
        and pair_results["object"]["margin_mean"] > 0.0
    )
    result = {
        "schema": "goal12-c1-counterfactual-offline-v1",
        "arm": args.arm,
        "checkpoint": {"path": str(args.checkpoint), "sha256": sha256_file(args.checkpoint)},
        "config": {"path": str(args.config), "sha256": sha256_file(args.config)},
        "sidecar": {"path": str(args.sidecar), "sha256": sha256_file(args.sidecar)},
        "point_manifest": (
            {"path": str(args.point_manifest), "sha256": sha256_file(args.point_manifest)}
            if args.point_manifest is not None
            else None
        ),
        "seed": args.seed,
        "num_groups": len(samples),
        "pair_results": pair_results,
        "region_results": region_results,
        "point_results": point_results,
        "gate": {
            "destination_margin_positive_with_ci": destination_passed,
            "object_margin_positive": object_passed,
            "region_predicted_action_beats_shuffle": bool(
                region_results
                and region_results["action_comparison"]["shuffle_minus_predicted"] > 0.0
                and region_results["action_comparison"][
                    "shuffle_minus_predicted_bootstrap_95ci"
                ][0]
                > 0.0
            ),
            "region_predicted_action_beats_none": bool(
                region_results
                and region_results["action_comparison"]["none_minus_predicted"] > 0.0
                and region_results["action_comparison"][
                    "none_minus_predicted_bootstrap_95ci"
                ][0]
                > 0.0
            ),
            "region_predicted_mask_beats_shuffle": bool(
                region_results
                and region_results["iou"]["destination_predicted"]
                > region_results["iou"]["destination_shuffle"]
            ),
            "region_predicted_mask_beats_constant": bool(
                region_results
                and region_results["iou"]["destination_predicted"]
                > region_results["iou"]["destination_constant"]
            ),
            "point_oracle_beats_none": bool(
                point_results
                and point_results["action_comparison"]["oracle_mae"]
                < point_results["action_comparison"]["none_mae"]
            ),
            "point_predicted_beats_shuffle": bool(
                point_results
                and point_results["action_comparison"]["predicted_minus_shuffle"] > 0.0
            ),
            "point_predicted_beats_none": bool(
                point_results
                and point_results["action_comparison"]["predicted_minus_none"] > 0.0
            ),
        },
        "notes": [
            "Pairs use the same image, frame and expert action chunk; only language changes.",
            "Correct and wrong calls for each subset use the same seed and batch order.",
            "white_right is not evaluated here because it has no expert action labels.",
            "Oracle point markers are diagnostic only and never enter deployment inputs.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
