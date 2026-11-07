#!/usr/bin/env python3
"""Train a P1-C single-step xy/release student on the frozen T4 cache.

The teacher labels come from P1-B.  Only the first action of each cache row is
supervised because P1-B provides a per-transition teacher delta, not a real
8-step teacher rollout.  ``R1-CAP`` zeroes route and geometry inputs,
``R2-ROUTE`` uses route tokens, and ``R3-GEO`` uses the deployable 6D route
geometry as an explicit-geometry diagnostic.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.k4_lite_t4_common import load_t4_cache, read_jsonl, sha256_file
from src.vla_attention.p1_student import P1Student, arm_inputs


THRESHOLDS = np.asarray([0.01, 0.03, 0.05, 0.10], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--train-teacher", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--validation-teacher", type=Path, required=True)
    parser.add_argument("--arm", choices=("R1-CAP", "R2-ROUTE", "R3-GEO"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda:0" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def load_split(cache_path: Path, teacher_path: Path, *, split: str) -> dict[str, Any]:
    cache = load_t4_cache(cache_path)
    summary = json.loads((teacher_path / "summary.json").read_text())
    teacher_split = summary.get("inventory_split")
    meta_splits = {str(row.get("split")) for row in cache["meta"]}
    if teacher_split is not None and teacher_split != split:
        raise ValueError(
            f"{teacher_path}: expected inventory_split={split}, got {teacher_split}"
        )
    if meta_splits != {split}:
        raise ValueError(f"{teacher_path}: expected meta split={split}, got {meta_splits}")
    source_index = np.load(teacher_path / "source_index.npy")
    delta = np.load(teacher_path / "teacher_delta_xy.npy").astype(np.float32)
    release = np.load(teacher_path / "teacher_release.npy").astype(np.float32)
    release_active = np.load(teacher_path / "teacher_release_active.npy").astype(np.float32)
    if source_index.ndim != 1 or delta.shape != (len(source_index), 2):
        raise ValueError(f"{teacher_path}: malformed teacher arrays")
    if release.shape != (len(source_index),) or release_active.shape != (len(source_index),):
        raise ValueError(f"{teacher_path}: malformed release arrays")
    if len(np.unique(source_index)) != len(source_index):
        raise ValueError(f"{teacher_path}: duplicated source_index")
    cache_count = len(cache["meta"])
    if source_index.min(initial=0) < 0 or source_index.max(initial=-1) >= cache_count:
        raise ValueError(f"{teacher_path}: source_index outside cache range")
    if len(cache["meta"]) < len(source_index):
        raise ValueError(f"{teacher_path}: teacher rows exceed cache rows")
    teacher_meta = read_jsonl(teacher_path / "meta.jsonl")
    if [int(row["source_index"]) for row in teacher_meta] != source_index.tolist():
        raise ValueError(f"{teacher_path}: teacher metadata order does not match source_index")
    if len(teacher_meta) != len(source_index):
        raise ValueError(f"{teacher_path}: metadata/teacher row mismatch")
    arrays = cache["arrays"]
    for name in ("z_source", "z_target", "z_global", "base_action", "proprio"):
        if arrays[name].shape[0] != cache_count:
            raise ValueError(f"{teacher_path}: cache {name} row mismatch")
    if not np.isfinite(delta).all() or not np.isfinite(release).all():
        raise ValueError(f"{teacher_path}: non-finite teacher labels")
    if not set(np.unique(release).tolist()).issubset({0.0, 1.0}):
        raise ValueError(f"{teacher_path}: release labels must be binary")
    ordered_meta = [cache["meta"][int(index)] for index in source_index]
    for row in ordered_meta:
        language = str(row.get("language", "")).lower()
        if "white mug" in language and "right plate" in language:
            raise ValueError(f"{teacher_path}: white_right found in {split} metadata")
    return {
        "path": cache_path,
        "teacher_path": teacher_path,
        "summary": summary,
        "meta": ordered_meta,
        "source_index": source_index,
        "z_source": np.asarray(arrays["z_source"][source_index], dtype=np.float32),
        "z_target": np.asarray(arrays["z_target"][source_index], dtype=np.float32),
        "z_global": np.asarray(arrays["z_global"][source_index], dtype=np.float32),
        "proprio": np.asarray(arrays["proprio"][source_index], dtype=np.float32),
        "base_action": np.asarray(arrays["base_action"][source_index], dtype=np.float32),
        "geometry": np.asarray(arrays["geometry"][source_index], dtype=np.float32),
        "delta": delta,
        "release": release,
        "release_active": release_active,
    }


def make_sample_weights(
    delta: np.ndarray,
    *,
    weight_min: float,
    weight_max: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if weight_min <= 0 or weight_max < weight_min:
        raise ValueError("invalid delta weight range")
    norms = np.linalg.norm(delta, axis=1)
    bins = np.digitize(norms, THRESHOLDS).astype(np.int64)
    counts = np.bincount(bins, minlength=5).astype(np.float64)
    if (counts == 0).any():
        raise ValueError(f"empty teacher delta bin: {counts.tolist()}")
    raw = len(bins) / (5.0 * counts[bins])
    raw = np.clip(raw, weight_min, weight_max)
    raw = raw / raw.mean()
    return raw.astype(np.float32), {
        "counts": counts.astype(int).tolist(),
        "fractions": (counts / counts.sum()).tolist(),
        "thresholds_m": THRESHOLDS.tolist(),
        "weight_min": float(weight_min),
        "weight_max": float(weight_max),
        "mean_weight": float(raw.mean()),
    }


def batch_to_torch(
    data: dict[str, Any], indices: np.ndarray, *, device: str
) -> dict[str, Any]:
    import torch

    def take(name: str):
        return torch.from_numpy(np.asarray(data[name][indices], dtype=np.float32)).to(device)

    return {
        "z_source": take("z_source"),
        "z_target": take("z_target"),
        "z_global": take("z_global"),
        "proprio": take("proprio"),
        "base_action": take("base_action"),
        "geometry": take("geometry"),
        "delta": take("delta"),
        "release_target": take("release_target"),
    }


def build_model(config: dict[str, Any], *, device: str) -> P1Student:
    model = P1Student(
        feature_dim=int(config["feature_dim"]),
        action_dim=int(config["action_dim"]),
        horizon=int(config["horizon"]),
        proprio_dim=int(config["proprio_dim"]),
        geometry_dim=int(config["geometry_dim"]),
        feature_bottleneck_dim=int(config["feature_bottleneck_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
        max_translation_delta=float(config["max_translation_delta"]),
        gate_mode=str(config.get("gate_mode", "tanh")),
        release_bias=float(config.get("release_bias", -4.0)),
    )
    return model.to(device)


def forward_batch(model: P1Student, batch: dict[str, Any], *, arm: str) -> dict[str, Any]:
    use_route, use_geometry = arm_inputs(arm)
    return model(
        batch["z_source"],
        batch["z_target"],
        batch["z_global"],
        batch["proprio"],
        batch["base_action"],
        geometry=batch["geometry"],
        use_route=use_route,
        use_geometry=use_geometry,
    )


def train(
    model: P1Student,
    data: dict[str, Any],
    weights: np.ndarray,
    *,
    arm: str,
    config: dict[str, Any],
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
    log_every: int,
) -> dict[str, Any]:
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(config["weight_decay"]),
    )
    rng = np.random.default_rng(seed)
    count = len(data["delta"])
    order = rng.permutation(count)
    cursor = 0
    pos_weight = torch.tensor(float(config["release_pos_weight"]), device=device)
    release_weight = float(config["release_loss_weight"])
    correction_weight = float(config.get("correction_loss_weight", 0.5))
    grad_clip = float(config["grad_clip_norm"])
    initial_loss = None
    final_loss = None
    logs: list[dict[str, Any]] = []
    started = time.time()
    for step in range(1, max_steps + 1):
        if cursor == 0 or cursor + batch_size > count:
            order = rng.permutation(count)
            cursor = 0
        indices = order[cursor : cursor + batch_size]
        cursor += batch_size
        batch = batch_to_torch(data, indices, device=device)
        out = forward_batch(model, batch, arm=arm)
        per_sample = torch.nn.functional.smooth_l1_loss(
            out["delta_xy"], batch["delta"], reduction="none"
        ).mean(dim=1)
        sample_weights = torch.from_numpy(weights[indices]).to(device)
        delta_loss = (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
        release_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out["release_logit"], batch["release_target"], pos_weight=pos_weight
        )
        correction_target = (torch.linalg.vector_norm(batch["delta"], dim=1) > 1e-8).float()
        correction_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out["correction_logit"], correction_target
        )
        loss = delta_loss + release_weight * release_loss + correction_weight * correction_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        value = float(loss.detach().cpu())
        if initial_loss is None:
            initial_loss = value
        final_loss = value
        if step == 1 or step % log_every == 0 or step == max_steps:
            logs.append(
                {
                    "step": step,
                    "loss": value,
                    "delta_loss": float(delta_loss.detach().cpu()),
                    "release_loss": float(release_loss.detach().cpu()),
                    "correction_loss": float(correction_loss.detach().cpu()),
                    "gate": float(out["gate"].detach().cpu()),
                }
            )
            print(json.dumps(logs[-1], sort_keys=True), flush=True)
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "logs": logs,
        "elapsed_seconds": time.time() - started,
    }


def evaluate(
    model: P1Student,
    data: dict[str, Any],
    *,
    arm: str,
    device: str,
    correction_threshold: float = 0.5,
) -> dict[str, Any]:
    import torch

    model.eval()
    preds: list[np.ndarray] = []
    release_probs: list[np.ndarray] = []
    correction_probs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(data["delta"]), 256):
            indices = np.arange(start, min(start + 256, len(data["delta"])))
            batch = batch_to_torch(data, indices, device=device)
            out = forward_batch(model, batch, arm=arm)
            active = (
                torch.sigmoid(out["correction_logit"])
                >= float(correction_threshold)
            ).float().reshape(-1, 1)
            preds.append((out["delta_xy"] * active).cpu().numpy())
            release_probs.append(torch.sigmoid(out["release_logit"]).cpu().numpy())
            correction_probs.append(
                torch.sigmoid(out["correction_logit"]).cpu().numpy()
            )
    return {
        "delta": np.concatenate(preds, axis=0),
        "release_prob": np.concatenate(release_probs, axis=0),
        "correction_prob": np.concatenate(correction_probs, axis=0),
    }


def metrics(
    target: np.ndarray,
    pred: np.ndarray,
    release_target: np.ndarray,
    release_prob: np.ndarray,
    *,
    release_threshold: float,
    correction_target: np.ndarray | None = None,
    correction_prob: np.ndarray | None = None,
    correction_threshold: float = 0.5,
) -> dict[str, Any]:
    target_norm = np.linalg.norm(target, axis=1)
    pred_norm = np.linalg.norm(pred, axis=1)
    nonzero = target_norm > 1e-8
    cosine = np.einsum("ij,ij->i", pred, target) / (pred_norm * target_norm + 1e-12)
    target_axis = np.argmax(np.abs(target), axis=1)
    rows = np.arange(len(target))
    sign_ok = np.sign(pred[rows, target_axis]) == np.sign(target[rows, target_axis])
    magnitude_ratio = pred_norm / (target_norm + 1e-12)
    abs_error = np.abs(pred - target).mean(axis=1)
    bins = np.digitize(target_norm, THRESHOLDS).astype(np.int64)
    per_bin: dict[str, Any] = {}
    for value in range(5):
        mask = bins == value
        per_bin[f"bin_{value}"] = {
            "count": int(mask.sum()),
            "mae": float(abs_error[mask].mean()) if mask.any() else None,
            "cosine_mean": float(cosine[mask & nonzero].mean()) if (mask & nonzero).any() else None,
            "sign_accuracy": float(sign_ok[mask & nonzero].mean()) if (mask & nonzero).any() else None,
        }
    predicted_release = release_prob >= release_threshold
    actual_release = release_target >= 0.5
    tp = int((predicted_release & actual_release).sum())
    fp = int((predicted_release & ~actual_release).sum())
    fn = int((~predicted_release & actual_release).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    correction_info = None
    if correction_target is not None and correction_prob is not None:
        correction_pred = correction_prob >= correction_threshold
        correction_actual = correction_target >= 0.5
        ctp = int((correction_pred & correction_actual).sum())
        cfp = int((correction_pred & ~correction_actual).sum())
        cfn = int((~correction_pred & correction_actual).sum())
        correction_info = {
            "threshold": float(correction_threshold),
            "positive_count": int(correction_actual.sum()),
            "predicted_positive_count": int(correction_pred.sum()),
            "tp": ctp,
            "fp": cfp,
            "fn": cfn,
            "precision": float(ctp / (ctp + cfp)) if ctp + cfp else 0.0,
            "recall": float(ctp / (ctp + cfn)) if ctp + cfn else 0.0,
        }
    plateau = (target_norm >= 0.027) & (target_norm <= 0.035)
    large = (target_norm >= 0.05) & (target_norm <= 0.10)
    nonzero_mae = float(abs_error[nonzero].mean()) if nonzero.any() else float("nan")
    checks = {
        "cosine_mean_ge_0.50": bool(cosine[nonzero].mean() >= 0.50) if nonzero.any() else False,
        "dominant_axis_sign_ge_0.75": bool(sign_ok[nonzero].mean() >= 0.75) if nonzero.any() else False,
        "magnitude_ratio_median_0.70_1.30": bool(
            0.70 <= np.median(magnitude_ratio[nonzero]) <= 1.30
        ) if nonzero.any() else False,
        "plateau_bucket_not_worse_than_nonzero_1.25x": bool(
            plateau.any() and abs_error[plateau].mean() <= 1.25 * nonzero_mae
        ),
        "large_bucket_not_worse_than_nonzero_1.25x": bool(
            large.any() and abs_error[large].mean() <= 1.25 * nonzero_mae
        ),
        "release_positive_available": bool(actual_release.any()),
        "release_precision_ge_0.70": bool(precision >= 0.70),
        "release_recall_ge_0.60": bool(recall >= 0.60),
        "residual_xy_abs_max_le_0.10": bool(np.abs(pred).max() <= 0.10 + 1e-6),
        "z_rotation_zero": True,
        "finite": bool(np.isfinite(pred).all() and np.isfinite(release_prob).all()),
    }
    return {
        "n": int(len(target)),
        "target_nonzero_count": int(nonzero.sum()),
        "target_nonzero_fraction": float(nonzero.mean()),
        "target_zero_fraction": float((~nonzero).mean()),
        "pred_zero_fraction": float((pred_norm <= 1e-8).mean()),
        "pred_any_axis_cap_fraction": float((np.abs(pred) >= 0.099999).any(axis=1).mean()),
        "delta_mae_all": float(abs_error.mean()),
        "delta_mae_nonzero": nonzero_mae,
        "cosine_mean_nonzero": float(cosine[nonzero].mean()) if nonzero.any() else None,
        "cosine_positive_fraction_nonzero": float((cosine[nonzero] > 0).mean()) if nonzero.any() else None,
        "dominant_axis_sign_accuracy_nonzero": float(sign_ok[nonzero].mean()) if nonzero.any() else None,
        "magnitude_ratio_median_nonzero": float(np.median(magnitude_ratio[nonzero])) if nonzero.any() else None,
        "per_bin": per_bin,
        "plateau": {
            "count": int(plateau.sum()),
            "mae": float(abs_error[plateau].mean()) if plateau.any() else None,
        },
        "large": {
            "count": int(large.sum()),
            "mae": float(abs_error[large].mean()) if large.any() else None,
        },
        "release": {
            "threshold": float(release_threshold),
            "positive_count": int(actual_release.sum()),
            "predicted_positive_count": int(predicted_release.sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        },
        "correction_gate": correction_info,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def main() -> int:
    args = parse_args()
    import torch

    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    if float(config.get("max_z_delta", 0.0)) != 0.0:
        raise ValueError("P1-C requires max_z_delta=0")
    if float(config.get("max_rotation_delta", 0.0)) != 0.0:
        raise ValueError("P1-C requires max_rotation_delta=0")
    if float(config["max_translation_delta"]) != 0.10:
        raise ValueError("P1-C requires max_translation_delta=0.10")
    max_steps = int(config["smoke_steps"] if args.max_steps is None else args.max_steps)
    batch_size = int(config["batch_size"] if args.batch_size is None else args.batch_size)
    learning_rate = float(
        config["learning_rate"] if args.learning_rate is None else args.learning_rate
    )
    seed = int(config["seed"] if args.seed is None else args.seed)
    if max_steps <= 0 or batch_size <= 0:
        raise ValueError("max_steps and batch_size must be positive")
    torch.set_float32_matmul_precision("high")
    train_data = load_split(args.train_cache, args.train_teacher, split="train")
    validation_data = load_split(args.validation_cache, args.validation_teacher, split="validation")
    release_target_kind = str(config.get("release_target", "teacher_release_active"))
    if release_target_kind not in ("teacher_release", "teacher_release_active"):
        raise ValueError(f"unsupported release_target: {release_target_kind}")
    for data in (train_data, validation_data):
        data["release_target"] = (
            data["release"]
            if release_target_kind == "teacher_release"
            else data["release_active"]
        )
    weights, weight_summary = make_sample_weights(
        train_data["delta"],
        weight_min=float(config["delta_weight_min"]),
        weight_max=float(config["delta_weight_max"]),
    )
    model = build_model(config, device=args.device)
    init_batch = batch_to_torch(train_data, np.arange(min(8, len(train_data["delta"]))), device=args.device)
    with torch.inference_mode():
        init_out = forward_batch(model, init_batch, arm=args.arm)
        init_identity_max = float(init_out["delta_xy"].abs().max().cpu())
    train_record = train(
        model,
        train_data,
        weights,
        arm=args.arm,
        config=config,
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        device=args.device,
        log_every=args.log_every,
    )
    correction_threshold = float(config.get("correction_threshold", 0.5))
    validation_pred = evaluate(
        model,
        validation_data,
        arm=args.arm,
        device=args.device,
        correction_threshold=correction_threshold,
    )
    validation_metrics = metrics(
        validation_data["delta"],
        validation_pred["delta"],
        validation_data["release_target"],
        validation_pred["release_prob"],
        release_threshold=float(config["release_threshold"]),
        correction_target=(np.linalg.norm(validation_data["delta"], axis=1) > 1e-8).astype(np.float32),
        correction_prob=validation_pred["correction_prob"],
        correction_threshold=correction_threshold,
    )
    train_pred = evaluate(
        model,
        train_data,
        arm=args.arm,
        device=args.device,
        correction_threshold=correction_threshold,
    )
    train_metrics = metrics(
        train_data["delta"],
        train_pred["delta"],
        train_data["release_target"],
        train_pred["release_prob"],
        release_threshold=float(config["release_threshold"]),
        correction_target=(np.linalg.norm(train_data["delta"], axis=1) > 1e-8).astype(np.float32),
        correction_prob=train_pred["correction_prob"],
        correction_threshold=correction_threshold,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    args.output.mkdir(parents=True)
    torch.save({"state_dict": model.state_dict(), "config": config, "arm": args.arm}, args.output / "student.pt")
    np.savez_compressed(
        args.output / "validation_predictions.npz",
        target_delta=validation_data["delta"],
        pred_delta=validation_pred["delta"],
        release_target=validation_data["release_target"],
        release_prob=validation_pred["release_prob"],
        correction_target=(np.linalg.norm(validation_data["delta"], axis=1) > 1e-8).astype(np.float32),
        correction_prob=validation_pred["correction_prob"],
    )
    summary = {
        "status": "PASS" if validation_metrics["passed"] else "FAIL",
        "stage": "P1-C",
        "arm": args.arm,
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "train_script": str(Path(__file__).resolve()),
        "train_script_sha256": sha256_file(Path(__file__).resolve()),
        "student_module": str((ROOT / "src/vla_attention/p1_student.py").resolve()),
        "student_module_sha256": sha256_file(ROOT / "src/vla_attention/p1_student.py"),
        "train_cache": str(args.train_cache),
        "train_cache_summary_sha256": sha256_file(args.train_cache / "summary.json"),
        "train_teacher": str(args.train_teacher),
        "train_teacher_summary_sha256": sha256_file(args.train_teacher / "summary.json"),
        "validation_cache": str(args.validation_cache),
        "validation_cache_summary_sha256": sha256_file(args.validation_cache / "summary.json"),
        "validation_teacher": str(args.validation_teacher),
        "validation_teacher_summary_sha256": sha256_file(args.validation_teacher / "summary.json"),
        "student": "P1Student",
        "state_source": "hdf5_expert_replay",
        "supervision": f"single_step_xy_plus_{release_target_kind}",
        "release_target": release_target_kind,
        "release_pos_weight": float(config["release_pos_weight"]),
        "correction_loss_weight": float(config.get("correction_loss_weight", 0.5)),
        "correction_threshold": correction_threshold,
        "teacher_delta_chunk_used": False,
        "geometry_kind": "route_centroid_6d" if args.arm == "R3-GEO" else "disabled",
        "max_steps": max_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "parameter_count": int(parameter_count),
        "trainable_parameter_count": int(trainable_parameter_count),
        "init_identity_max": init_identity_max,
        "sample_weights": weight_summary,
        "train_record": train_record,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "checkpoint_sha256": sha256_file(args.output / "student.pt"),
        "notes": [
            "Only the first action step is supervised; teacher_delta_chunk is not used.",
            "z and rotation have no output path and are exactly zero.",
            "R3-GEO uses deployable route-centroid geometry as an explicit-geometry diagnostic, not simulator GT.",
            "No closed-loop rollout or 15-init evaluation was run.",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    lines = [
        f"# P1-C {args.arm}",
        "",
        f"- status: `{summary['status']}`",
        f"- steps: `{max_steps}`",
        f"- validation nonzero targets: `{validation_metrics['target_nonzero_count']}`",
        f"- validation cosine: `{validation_metrics['cosine_mean_nonzero']}`",
        f"- validation sign accuracy: `{validation_metrics['dominant_axis_sign_accuracy_nonzero']}`",
        f"- validation magnitude ratio median: `{validation_metrics['magnitude_ratio_median_nonzero']}`",
        f"- validation release precision/recall: `{validation_metrics['release']['precision']:.4f}` / `{validation_metrics['release']['recall']:.4f}`",
        f"- failed checks: `{[name for name, value in validation_metrics['checks'].items() if not value]}`",
    ]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if validation_metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
