"""Produce paired, deterministic F0 directionality metrics and analysis text."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np


def load(path: Path):
    return json.loads(path.read_text())


def paired(a, b, key, seed=23, draws=10000):
    x = np.asarray([row["metrics"][key] for row in a["rows"]], dtype=float)
    y = np.asarray([row["metrics"][key] for row in b["rows"]], dtype=float)
    d = x - y
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), size=(draws, len(d)))].mean(axis=1)
    return {"mean_delta": float(d.mean()), "median_delta": float(np.median(d)),
            "positive_fraction": float((d > 0).mean()),
            "bootstrap_ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True); p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    root = a.root
    reports = {name: load(root / filename) for name, filename in {
        "V0_20": "V0_heldout_report.json", "V1_20": "V1_heldout_report.json",
        "V2_20": "V2_heldout_report.json", "V3_20": "V3_heldout_report.json",
        "V4_20": "V4_heldout_report.json", "V1_100": "V1_100step_heldout_report.json",
        "V2_100": "V2_100step_heldout_report.json", "V3_100": "V3_100step_heldout_report.json",
        "V4_100": "V4_100step_heldout_report.json"}.items()}
    keys = ("pointing", "mass_in_box", "top20_box_iou")
    metrics = {name: data["summary"] for name, data in reports.items()}
    metrics["paired_100_V3_minus_V1"] = {k: paired(reports["V3_100"], reports["V1_100"], k) for k in keys}
    metrics["paired_100_V4_minus_V2"] = {k: paired(reports["V4_100"], reports["V2_100"], k) for k in keys}
    metrics["paired_20_V3_minus_V1"] = {k: paired(reports["V3_20"], reports["V1_20"], k) for k in keys}
    metrics["paired_20_V4_minus_V2"] = {k: paired(reports["V4_20"], reports["V2_20"], k) for k in keys}
    teacher = load(root / "teacher_controls.json")["summary"]
    metrics["teacher_controls"] = teacher
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(metrics, indent=2) + "\n")
    lines = ["# F0 idea validation analysis", "", "## Scope", "",
             "64 image-disjoint Flickr30k Entities test records; phrase-score gradient×activation; same processor and evaluation code. 20-step is an engineering smoke; 100-step is the directionality check.", "",
             "## Results", "", "| Run | pointing | mass-in-box | top20 box IoU |", "|---|---:|---:|---:|"]
    for name in ("V0_20", "V1_20", "V2_20", "V3_20", "V4_20", "V1_100", "V2_100", "V3_100", "V4_100"):
        d = metrics[name]; lines.append(f"| {name} | {d['pointing']:.4f} | {d['mass_in_box']:.4f} | {d['top20_box_iou']:.4f} |")
    lines += ["", "## Paired 100-step deltas", "", "V3 − V1: pointing improves, while mass-in-box and IoU do not show a positive mean delta. V4 − V2: mass-in-box and IoU improve, while pointing is not improved.", "",
              "Bootstrap intervals use 10,000 paired resamples with seed 23. These are directional intervals, not a final large-scale significance claim.", "",
              "## Teacher controls", "", "Correct SD phrase maps outperform wrong-image and random controls on pointing and mass-in-box. The shifted control remains close to correct on IoU, so the current IoU negative control is not clean enough to support a strong causal claim.", "",
              "## Decision", "", "The 100-step run provides partial evidence: semantic attribution alignment can improve some spatial metrics, especially when combined with retention, but the full V3/V4 success criterion is not met. Do not launch F1-10k unchanged. First revise the loss strength/schedule or teacher-map evaluation, add a clean negative control, and rerun a small controlled sweep."]
    (a.output.parent / "analysis.md").write_text("\n".join(lines) + "\n")
    print(a.output)


if __name__ == "__main__": main()
