"""Calibrate map resolution and support thresholds on a non-test split.

This script never changes checkpoints. It compares saved student attribution
maps and frozen teacher maps on a declared calibration manifest, then writes
the selected common resolution and support rule for later held-out evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def resize_bilinear(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D map, got {arr.shape}")
    sy = np.linspace(0, arr.shape[0] - 1, height)
    sx = np.linspace(0, arr.shape[1] - 1, width)
    y0 = np.floor(sy).astype(int); y1 = np.minimum(y0 + 1, arr.shape[0] - 1)
    x0 = np.floor(sx).astype(int); x1 = np.minimum(x0 + 1, arr.shape[1] - 1)
    wy = sy - y0; wx = sx - x0
    top = (1.0 - wx)[None, :] * arr[y0[:, None], x0[None, :]] + wx[None, :] * arr[y0[:, None], x1[None, :]]
    bot = (1.0 - wx)[None, :] * arr[y1[:, None], x0[None, :]] + wx[None, :] * arr[y1[:, None], x1[None, :]]
    return (1.0 - wy)[:, None] * top + wy[:, None] * bot


def normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.maximum(np.asarray(arr, dtype=np.float64), 0.0)
    total = float(arr.sum())
    return arr / total if total > 1e-12 else np.full_like(arr, 1.0 / arr.size)


def box_iou(a, b):
    l, t, r, bot = a; ll, tt, rr, bb = b
    inter = max(0.0, min(r, rr) - max(l, ll)) * max(0.0, min(bot, bb) - max(t, tt))
    union = max(0.0, r - l) * max(0.0, bot - t) + max(0.0, rr - ll) * max(0.0, bb - tt) - inter
    return inter / union if union > 0 else 0.0


def support_box(grid, fraction, width, height):
    flat = grid.reshape(-1)
    k = max(1, int(np.ceil(flat.size * fraction)))
    ys, xs = np.unravel_index(np.argsort(flat)[-k:], grid.shape)
    return (float(xs.min()) * width / grid.shape[1], float(ys.min()) * height / grid.shape[0],
            float(xs.max() + 1) * width / grid.shape[1], float(ys.max() + 1) * height / grid.shape[0])


def threshold_box(grid, quantile, width, height):
    threshold = float(np.quantile(grid, quantile))
    ys, xs = np.where(grid >= threshold)
    if len(xs) == 0:
        return support_box(grid, 0.20, width, height)
    return (float(xs.min()) * width / grid.shape[1], float(ys.min()) * height / grid.shape[0],
            float(xs.max() + 1) * width / grid.shape[1], float(ys.max() + 1) * height / grid.shape[0])


def gt_mask(boxes, width, height, h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    px = (xx + 0.5) * width / w; py = (yy + 0.5) * height / h
    mask = np.zeros((h, w), dtype=bool)
    for left, top, right, bottom in boxes:
        mask |= (px >= left) & (px <= right) & (py >= top) & (py <= bottom)
    return mask


def sample_metrics(grid, boxes, width, height, fractions, thresholds):
    target = gt_mask(boxes, width, height, *grid.shape)
    soft = (grid - grid.min()) / max(float(grid.max() - grid.min()), 1e-12)
    soft_iou = float(np.minimum(soft, target).sum() / max(np.maximum(soft, target).sum(), 1e-12))
    out = {"soft_iou": soft_iou, "mass_in_box": float(grid[target].sum())}
    for f in fractions:
        pred = support_box(grid, f, width, height)
        out[f"box_iou_top{int(round(100*f))}"] = max(box_iou(pred, b) for b in boxes)
    for q in thresholds:
        pred = threshold_box(grid, q, width, height)
        out[f"box_iou_q{int(round(100*q))}"] = max(box_iou(pred, b) for b in boxes)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student-report", type=Path, required=True)
    p.add_argument("--teacher-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resolutions", default="16,32,64")
    p.add_argument("--fractions", default="0.1,0.2,0.3,0.4")
    p.add_argument("--thresholds", default="0.5,0.7,0.8,0.9,0.95")
    p.add_argument("--model-id", default="student")
    a = p.parse_args()
    report = json.loads(a.student_report.read_text())
    manifest = {row["sample_id"]: row for row in (json.loads(x) for x in a.manifest.read_text().splitlines() if x.strip())}
    resolutions = [int(x) for x in a.resolutions.split(",")]
    fractions = [float(x) for x in a.fractions.split(",")]
    thresholds = [float(x) for x in a.thresholds.split(",")]
    results = []
    for resolution in resolutions:
        acc = []
        for row in report["rows"]:
            sample = manifest[row["sample_id"]]
            student = np.load(row["map_path"])
            teacher_path = a.teacher_dir / (row["sample_id"].replace(":", "_") + ".npy")
            teacher = np.load(teacher_path)
            if teacher.ndim != 2:
                teacher = teacher.reshape(int(round(np.sqrt(teacher.size))), -1)
            student = normalize(resize_bilinear(student, resolution, resolution))
            teacher = normalize(resize_bilinear(teacher, resolution, resolution))
            width, height = row.get("image_size", sample.get("image_size", [1, 1]))
            acc.append({"sample_id": row["sample_id"], "student": sample_metrics(student, sample["boxes_xyxy"], width, height, fractions, thresholds), "teacher": sample_metrics(teacher, sample["boxes_xyxy"], width, height, fractions, thresholds)})
        summary = {}
        for side in ("student", "teacher"):
            keys = acc[0][side].keys()
            summary[side] = {key: float(np.mean([r[side][key] for r in acc])) for key in keys}
        results.append({"resolution": resolution, "summary": summary, "n": len(acc)})
    # Calibration criterion: maximize student soft-IoU, then top-20 IoU.
    best = max(results, key=lambda r: (r["summary"]["student"]["soft_iou"], r["summary"]["student"].get("box_iou_top20", 0.0)))
    best_threshold = max(thresholds, key=lambda q: max(r["summary"]["student"].get(f"box_iou_q{int(round(100*q))}", 0.0) for r in results))
    payload = {"model_id": a.model_id, "manifest": str(a.manifest), "student_report": str(a.student_report), "teacher_dir": str(a.teacher_dir), "calibration_only": True, "resolutions": results, "selected_resolution": best["resolution"], "selected_threshold_quantile": best_threshold, "fractions": fractions, "thresholds": thresholds}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"selected_resolution": best["resolution"], "selected_threshold_quantile": best_threshold, "n": best["n"]}, indent=2))


if __name__ == "__main__":
    main()
