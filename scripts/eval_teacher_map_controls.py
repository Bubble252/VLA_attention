"""Evaluate correct and deliberately corrupted teacher maps on held-out rows.

This isolates whether a gain can be explained by an arbitrary spatial prior.
The script never trains a model: it scores cached maps against the same boxes,
then creates wrong-image, deterministic random, and spatially shifted controls.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np

from vla_attention.evaluation.spatial import mass_in_boxes, pointing_correct


def top20_iou(grid, boxes, width, height):
    flat = grid.reshape(-1)
    k = max(1, int(np.ceil(flat.size * .2)))
    ys, xs = np.unravel_index(np.argsort(flat)[-k:], grid.shape)
    l, r = xs.min() * width / grid.shape[1], (xs.max() + 1) * width / grid.shape[1]
    t, b = ys.min() * height / grid.shape[0], (ys.max() + 1) * height / grid.shape[0]
    best = 0.0
    for bl, bt, br, bb in boxes:
        il, it, ir, ib = max(l, bl), max(t, bt), min(r, br), min(b, bb)
        inter = max(0, ir - il) * max(0, ib - it)
        union = (r - l) * (b - t) + (br - bl) * (bb - bt) - inter
        best = max(best, inter / union if union else 0.0)
    return best


def score(grid, row):
    from PIL import Image
    with Image.open(row["image_path"]) as image:
        width, height = image.size
    boxes = row["boxes_xyxy"]
    grid = np.maximum(np.asarray(grid, dtype=np.float64), 0)
    grid /= max(grid.sum(), 1e-12)
    return {
        "pointing": float(pointing_correct(grid, boxes, image_width=width, image_height=height)),
        "mass_in_box": float(mass_in_boxes(grid, boxes, image_width=width, image_height=height)),
        "top20_box_iou": float(top20_iou(grid, boxes, width, height)),
    }


def mean(rows):
    return {k: float(np.mean([x[k] for x in rows])) for k in rows[0]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=23)
    a = p.parse_args()
    rows = [json.loads(line) for line in a.manifest.read_text().splitlines() if line.strip()]
    for row in rows:
        row["image_path"] = str(a.dataset_root / row["image_path"])
    maps = [np.load(a.cache / (row["sample_id"].replace(":", "_") + ".npy")) for row in rows]
    by_image = {}
    for index, row in enumerate(rows):
        by_image.setdefault(row.get("image_id"), []).append(index)
    wrong_word = []
    for index, row in enumerate(rows):
        peers = [j for j in by_image.get(row.get("image_id"), []) if j != index]
        wrong_word.append(maps[peers[0] if peers else (index + 1) % len(maps)])
    rng = np.random.default_rng(a.seed)
    controls = {"correct": maps, "wrong_word_same_image": wrong_word, "wrong_image": maps[1:] + maps[:1], "shifted": [np.roll(x, 3, axis=1) for x in maps],
                "random": [rng.random(x.shape) for x in maps]}
    summary = {name: mean([score(grid, row) for grid, row in zip(values, rows)]) for name, values in controls.items()}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({"n": len(rows), "seed": a.seed, "summary": summary}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
