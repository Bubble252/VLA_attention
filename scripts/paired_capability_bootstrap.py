"""Paired bootstrap over identical capability-evaluation sample IDs."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.eval_capability_suite import binary_label, parse_choice, vqa_consensus


def read_report(path: Path):
    report = json.loads(path.read_text())
    records = report.get("records")
    if not records:
        raise ValueError(f"report has no per-sample records: {path}")
    keyed = {row["id"]: row for row in records}
    if len(keyed) != len(records):
        raise ValueError(f"duplicate report IDs: {path}")
    return keyed


def row_score(task: str, row: dict) -> float:
    prediction = str(row["prediction"])
    if task in ("vqa", "textvqa"):
        return vqa_consensus(prediction, row["answers"])
    if task in ("pope", "binary"):
        return float(binary_label(prediction) == str(row["label"]).lower())
    if task in ("multiple_choice", "worldmedqa"):
        choice = parse_choice(prediction, row.get("choices", []))
        return float(choice == str(row["answer"]).strip().upper())
    if task == "exact_match":
        norm = lambda x: " ".join(str(x).lower().strip().split())
        return float(norm(prediction) in {norm(x) for x in row["answers"]})
    raise ValueError(f"row-level bootstrap not supported for {task}")


def make_caption_scorers(scorer_path: str | None):
    if scorer_path:
        sys.path.append(scorer_path)
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge
    scorers = [("Bleu", Bleu(4)), ("ROUGE_L", Rouge()), ("CIDEr", Cider())]
    if shutil.which("java"):
        try:
            from pycocoevalcap.meteor.meteor import Meteor
            scorers.append(("METEOR", Meteor()))
        except (FileNotFoundError, OSError):
            pass
    return scorers


def caption_scores(rows_a, rows_b, ids: list[str], scorers):
    refs = {str(i): rows_a[i]["references"] for i in ids}
    hyps = {str(i): [rows_b[i]["prediction"]] for i in ids}
    result = {}
    for name, scorer in scorers:
        score, _ = scorer.compute_score(refs, hyps)
        if isinstance(score, (list, tuple)):
            for idx, value in enumerate(score, 1):
                result[f"{name}_{idx}"] = float(value)
        else:
            result[name] = float(score)
    return result


def bootstrap(rows_a, rows_b, task, ids, samples, seed, scorer_path=None):
    rng = np.random.default_rng(seed)
    if task == "mme_pair":
        by_pair = defaultdict(lambda: [[], []])
        for side, mapping in enumerate((rows_a, rows_b)):
            for sample_id in ids:
                row = mapping[sample_id]
                by_pair[str(row["pair_id"])][side].append(
                    binary_label(str(row["prediction"])) == str(row["label"]).lower())
        if any(len(a) != 2 or len(b) != 2 for a, b in by_pair.values()):
            raise ValueError("MME requires two predictions per pair for both checkpoints")
        units = sorted(by_pair)
        scores_a = np.asarray([all(by_pair[k][0]) for k in units], dtype=float)
        scores_b = np.asarray([all(by_pair[k][1]) for k in units], dtype=float)
        observed = float((scores_b - scores_a).mean())
        reps = np.empty(samples)
        for i in range(samples):
            draw = rng.integers(0, len(units), len(units))
            reps[i] = (scores_b[draw] - scores_a[draw]).mean()
        return {"pair_count": len(units), "pair_accuracy_baseline": float(scores_a.mean()),
                "pair_accuracy_candidate": float(scores_b.mean()),
                "pair_accuracy_delta": observed,
                "ci95": [float(np.quantile(reps, .025)), float(np.quantile(reps, .975))]}
    if task == "caption":
        scorers = make_caption_scorers(scorer_path)
        a_scores = caption_scores(rows_a, rows_a, ids, scorers)
        b_scores = caption_scores(rows_a, rows_b, ids, scorers)
        result = {}
        for key in a_scores:
            deltas = []
            for _ in range(samples):
                draw = rng.integers(0, len(ids), len(ids)); sampled = [ids[i] for i in draw]
                aa = caption_scores(rows_a, rows_a, sampled, scorers)[key]
                bb = caption_scores(rows_a, rows_b, sampled, scorers)[key]
                deltas.append(bb - aa)
            result[key] = {"baseline": a_scores[key], "candidate": b_scores[key],
                           "delta": b_scores[key] - a_scores[key],
                           "ci95": [float(np.quantile(deltas, .025)), float(np.quantile(deltas, .975))]}
        return {"n": len(ids), "metrics": result,
                "metric_note": "corpus-level bootstrap; METEOR is omitted when Java is unavailable"}
    a = np.asarray([row_score(task, rows_a[x]) for x in ids], dtype=float)
    b = np.asarray([row_score(task, rows_b[x]) for x in ids], dtype=float)
    d = b - a
    reps = np.empty(samples)
    for i in range(samples):
        draw = rng.integers(0, len(ids), len(ids)); reps[i] = d[draw].mean()
    return {"n": len(ids), "baseline_score": float(a.mean()), "candidate_score": float(b.mean()),
            "delta": float(d.mean()), "ci95": [float(np.quantile(reps, .025)), float(np.quantile(reps, .975))]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--pycocoevalcap-path")
    a = p.parse_args()
    base, candidate = read_report(a.baseline), read_report(a.candidate)
    if set(base) != set(candidate):
        raise ValueError(f"sample ID sets differ: {len(base)} vs {len(candidate)}")
    ids = sorted(base)
    tasks = {row["task"] for row in base.values()} | {row["task"] for row in candidate.values()}
    if len(tasks) != 1:
        raise ValueError(f"reports mix or mismatch tasks: {tasks}")
    task = next(iter(tasks))
    result = {"baseline": str(a.baseline), "candidate": str(a.candidate), "task": task,
              "bootstrap_samples": a.bootstrap, "seed": a.seed,
              "resampling": "paired sample IDs; conditional on the selected checkpoints",
              "comparison": bootstrap(base, candidate, task, ids, a.bootstrap, a.seed, a.pycocoevalcap_path)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["comparison"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
