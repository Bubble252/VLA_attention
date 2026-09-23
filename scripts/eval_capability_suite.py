"""Generate and score normalized Lavender-style VLM capability manifests.

Input is JSONL with one record per image-question/task item. This keeps model
inference identical across benchmarks; official dataset adapters are responsible
for converting source data into the documented manifest fields.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import string
import sys
import time
from collections import defaultdict
from pathlib import Path


_NUMBER_MAP = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
               "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
               "nine": "9", "ten": "10"}
_CONTRACTIONS = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
                 "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
                 "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd",
                 "hes": "he's", "id": "i'd", "ill": "i'll", "im": "i'm", "ive": "i've",
                 "isnt": "isn't", "itd": "it'd", "itll": "it'll", "its": "it's", "lets": "let's",
                 "mightnt": "mightn't", "mustnt": "mustn't", "shant": "shan't", "shed": "she'd",
                 "shes": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "thats": "that's",
                 "thered": "there'd", "therere": "there're", "theres": "there's", "theyd": "they'd",
                 "theyll": "they'll", "theyre": "they're", "theyve": "they've", "wasnt": "wasn't",
                 "wed": "we'd", "were": "we're", "weve": "we've", "werent": "weren't",
                 "whatll": "what'll", "whatre": "what're", "whats": "what's", "whatve": "what've",
                 "whered": "where'd", "wheres": "where's", "whos": "who's", "whove": "who've",
                 "wouldve": "would've", "wouldnt": "wouldn't", "youd": "you'd", "youll": "you'll",
                 "youre": "you're", "youve": "you've"}


def normalize_answer(value: str) -> str:
    text = value.lower().strip()
    # VQA evaluation normalization: handle commas/periods, punctuation, number
    # words, articles, and common contractions. This is close to the official
    # evaluator but remains independently implemented; final reporting should
    # still run the benchmark-provided scorer.
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    if re.search(r"\d\.\d", text) is None:
        text = text.replace(".", "")
    for punct in string.punctuation:
        if punct == "'":
            continue
        if punct in text and (" " + punct in text or punct + " " in text):
            text = text.replace(punct, " ")
        else:
            text = text.replace(punct, "")
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(_NUMBER_MAP.get(word, word) for word in text.split())
    text = " ".join(_CONTRACTIONS.get(word, word) for word in text.split())
    return " ".join(text.split())


def vqa_consensus(prediction: str, references: list[str]) -> float:
    """VQA consensus score using the official leave-one-annotator-out rule."""
    pred = normalize_answer(prediction)
    refs = [normalize_answer(x) for x in references]
    n = len(refs)
    if n == 0:
        return 0.0
    # For each annotator, compute agreement against the other n-1 answers,
    # cap at 3 matches, then average across annotators (VQA evaluation rule).
    return sum(min(sum(x == pred for j, x in enumerate(refs) if j != i) / 3.0, 1.0)
               for i in range(n)) / n


def binary_label(text: str) -> str:
    text = normalize_answer(text)
    match = re.search(r"\b(yes|no)\b", text)
    if not match:
        return "unknown"
    return match.group(1)


def binary_metrics(predictions: list[str], labels: list[str]) -> dict:
    tp = sum(p == "yes" and y == "yes" for p, y in zip(predictions, labels))
    fp = sum(p == "yes" and y == "no" for p, y in zip(predictions, labels))
    fn = sum(p != "yes" and y == "yes" for p, y in zip(predictions, labels))
    tn = sum(p == "no" and y == "no" for p, y in zip(predictions, labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": (tp + tn) / max(len(labels), 1), "precision_yes": precision, "recall_yes": recall, "f1_yes": f1,
            "unknown_predictions": sum(x == "unknown" for x in predictions)}


def parse_choice(text: str, choices: list[str]) -> str:
    text = text.strip()
    explicit = re.search(r"\b(?:answer|option|choice)\s*(?:is|:|=)?\s*\(?([A-Z])\)?(?:\b|[.):])", text, re.I)
    if explicit and (not choices or ord(explicit.group(1).upper()) - ord("A") < len(choices)):
        return explicit.group(1).upper()
    leading = re.match(r"^\s*\(?([A-Z])\)?(?:[.):]|\s+-)\s*", text, re.I)
    if leading and (not choices or ord(leading.group(1).upper()) - ord("A") < len(choices)):
        return leading.group(1).upper()
    if not choices:
        match = re.search(r"\b([A-Z])\b", text.upper())
        return match.group(1) if match else ""
    if re.fullmatch(r"\s*\(?([A-Z])\)?\s*", text, re.I):
        letter = re.fullmatch(r"\s*\(?([A-Z])\)?\s*", text, re.I).group(1).upper()
        if ord(letter) - ord("A") < len(choices):
            return letter
    pred = normalize_answer(text)
    for idx, choice in enumerate(choices):
        if normalize_answer(choice) == pred or normalize_answer(choice) in pred:
            return chr(ord("A") + idx)
    return ""


def score_records(records: list[dict], predictions: list[str], *, caption_scorer=None) -> dict:
    if len(records) != len(predictions):
        raise ValueError(f"records/predictions length mismatch: {len(records)} != {len(predictions)}")
    grouped: dict[str, list[tuple[dict, str]]] = defaultdict(list)
    for row, pred in zip(records, predictions):
        grouped[row["task"]].append((row, pred))
    output = {}
    for task, pairs in grouped.items():
        if task == "caption":
            if caption_scorer is None:
                output[task] = {"count": len(pairs), "metrics_status": "pending: install pycocoevalcap for CIDEr/BLEU/METEOR/ROUGE-L"}
            else:
                refs = {str(i): row["references"] for i, (row, _) in enumerate(pairs)}
                hyps = {str(i): [pred] for i, (_, pred) in enumerate(pairs)}
                scores, _ = caption_scorer.compute_score(refs, hyps)
                output[task] = {"count": len(pairs), **{name: float(value) for name, value in scores.items()}}
        elif task in ("vqa", "textvqa"):
            values = [vqa_consensus(pred, row["answers"]) for row, pred in pairs]
            metric = "vqa_consensus_pilot" if task == "vqa" else "textvqa_consensus_pilot"
            output[task] = {"count": len(values), metric: sum(values) / max(len(values), 1),
                            "metric_note": "leave-one-annotator-out consensus with generic answer normalization; run official benchmark evaluator for final reporting"}
        elif task in ("pope", "binary"):
            labels = [row["label"].lower() for row, _ in pairs]
            pred_labels = [binary_label(pred) for _, pred in pairs]
            output[task] = {"count": len(labels), **binary_metrics(pred_labels, labels)}
            groups = defaultdict(lambda: [[], []])
            for (row, _), prediction, label in zip(pairs, pred_labels, labels):
                group = str(row.get("category", row.get("subset", "all")))
                groups[group][0].append(prediction); groups[group][1].append(label)
            if len(groups) > 1 or (groups and next(iter(groups)) != "all"):
                output[task]["by_category"] = {name: binary_metrics(v[0], v[1]) for name, v in groups.items()}
        elif task in ("multiple_choice", "worldmedqa"):
            correct = 0
            by_language = defaultdict(lambda: [0, 0])
            by_type = defaultdict(lambda: [0, 0])
            for row, pred in pairs:
                choice = parse_choice(pred, row.get("choices", []))
                ok = choice == str(row["answer"]).strip().upper()
                correct += int(ok)
                if row.get("language") is not None:
                    by_language[str(row["language"])][0] += int(ok); by_language[str(row["language"])][1] += 1
                if row.get("question_type") is not None:
                    by_type[str(row["question_type"])][0] += int(ok); by_type[str(row["question_type"])][1] += 1
            output[task] = {"count": len(pairs), "accuracy": correct / max(len(pairs), 1),
                            "accuracy_by_language": {k: v[0] / max(v[1], 1) for k, v in by_language.items()},
                            "accuracy_by_question_type": {k: v[0] / max(v[1], 1) for k, v in by_type.items()}}
        elif task == "exact_match":
            scores = [normalize_answer(pred) in {normalize_answer(x) for x in row["answers"]} for row, pred in pairs]
            output[task] = {"count": len(scores), "accuracy": sum(scores) / max(len(scores), 1)}
        elif task == "mme_pair":
            pair_correct = defaultdict(list)
            by_category = defaultdict(lambda: [0, 0])
            for row, pred in pairs:
                ok = binary_label(pred) == row["label"].lower()
                pair_correct[str(row["pair_id"])].append(ok)
                category = str(row.get("category", "unspecified"))
                by_category[category][0] += int(ok); by_category[category][1] += 1
            malformed = {key: len(values) for key, values in pair_correct.items() if len(values) != 2}
            if malformed:
                raise ValueError(f"MME requires complete 2-question pairs; malformed={list(malformed.items())[:5]}")
            output[task] = {"count": len(pairs), "pair_count": len(pair_correct),
                            "pair_accuracy": sum(all(v) for v in pair_correct.values()) / max(len(pair_correct), 1),
                            "mme_score_sum": sum(v[0] for v in by_category.values()),
                            "mme_score_by_category": {k: v[0] for k, v in by_category.items()},
                            "accuracy_by_category": {k: v[0] / max(v[1], 1) for k, v in by_category.items()}}
        else:
            raise ValueError(f"unsupported task={task!r}")
    return output


def load_model(model_path: Path, adapter: Path | None, *, max_image_pixels: int):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, use_fast=False,
                                              min_pixels=256 * 28 * 28, max_pixels=max_image_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation="eager"
    ).cuda().eval()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False).eval()
    return model, processor


def atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(str(temp), str(path))


def resume_identity(*, manifest_sha256: str, model_id: str, model: Path,
                    adapter: Path | None, max_new_tokens: int, model_revision: str,
                    max_image_pixels: int, ids: list[str]) -> dict:
    return {"manifest_sha256": manifest_sha256, "model_id": model_id,
            "model": str(model), "adapter": str(adapter) if adapter else None,
            "max_new_tokens": max_new_tokens, "model_revision": model_revision,
            "max_image_pixels": max_image_pixels, "ordered_ids": ids}


def generate(model, processor, dataset_root: Path, row: dict, *, max_new_tokens: int):
    import torch
    from qwen_vl_utils import process_vision_info
    content = []
    if row.get("image"):
        content.append({"type": "image", "image": str(dataset_root / row["image"])})
    content.append({"type": "text", "text": row["prompt"]})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    batch = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt")
    batch = {k: v.cuda() if hasattr(v, "cuda") else v for k, v in batch.items()}
    with torch.inference_mode():
        generated = model.generate(**batch, do_sample=False, max_new_tokens=max_new_tokens)
    prompt_len = batch["input_ids"].shape[1]
    return processor.tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--limit", type=int)
    p.add_argument("--caption-metrics", action="store_true")
    p.add_argument("--resume", action="store_true", help="resume from <output>.partial.json after verifying inputs")
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--model-revision", default="unspecified")
    p.add_argument("--max-image-pixels", type=int, default=1280 * 28 * 28)
    a = p.parse_args()
    raw_manifest = a.manifest.read_bytes()
    records = [json.loads(x) for x in raw_manifest.decode().splitlines() if x.strip()]
    if a.limit is not None:
        records = records[:a.limit]
    if not records:
        raise ValueError("manifest contains no records after applying --limit")
    if a.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")
    for idx, row in enumerate(records):
        required = {"id", "task", "prompt"} - set(row)
        if required:
            raise ValueError(f"manifest row {idx} missing {sorted(required)}")
        if row.get("image") and not (a.dataset_root / row["image"]).is_file():
            raise FileNotFoundError(a.dataset_root / row["image"])
    ids = [row["id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest contains duplicate IDs; run the manifest validator")
    identity = resume_identity(manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
                               model_id=a.model_id, model=a.model, adapter=a.adapter,
                               max_new_tokens=a.max_new_tokens, model_revision=a.model_revision,
                               max_image_pixels=a.max_image_pixels, ids=ids)
    partial_path = a.output.with_name(a.output.name + ".partial.json")
    predictions = []
    latencies = []
    if a.resume and partial_path.exists():
        partial = json.loads(partial_path.read_text())
        if partial.get("identity") != identity:
            raise ValueError("partial evaluation identity does not match this model/manifest/config")
        predictions = list(partial.get("predictions", []))
        latencies = list(partial.get("latency_seconds", []))
        if len(predictions) != len(latencies) or len(predictions) > len(records):
            raise ValueError("partial evaluation has inconsistent prediction/latency counts")
        print(f"Resuming at {len(predictions)}/{len(records)}", flush=True)
    elif partial_path.exists() and not a.resume:
        raise FileExistsError(f"partial result exists; pass --resume or move it: {partial_path}")
    if a.output.exists():
        raise FileExistsError(f"refusing to overwrite completed result: {a.output}")
    model, processor = load_model(a.model, a.adapter, max_image_pixels=a.max_image_pixels)
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for idx in range(len(predictions), len(records)):
        row = records[idx]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        item_started = time.perf_counter()
        prediction = generate(model, processor, a.dataset_root, row, max_new_tokens=a.max_new_tokens)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        predictions.append(prediction)
        latencies.append(time.perf_counter() - item_started)
        print(f"[{idx + 1}/{len(records)}] {row['id']}", flush=True)
        if len(predictions) % a.checkpoint_every == 0 or len(predictions) == len(records):
            atomic_json_write(partial_path, {"identity": identity, "predictions": predictions,
                                             "latency_seconds": latencies,
                                             "completed": len(predictions), "total": len(records)})
    scorer = None
    scorer_notes = []
    if a.caption_metrics and any(r["task"] == "caption" for r in records):
        # Package's public API expects COCO objects; use its standard scorers directly.
        scorer_path = os.environ.get("PYCOCOEVALCAP_PATH")
        if scorer_path:
            sys.path.append(scorer_path)
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.rouge.rouge import Rouge
        scorers = [("Bleu", Bleu(4)), ("ROUGE_L", Rouge()), ("CIDEr", Cider())]
        try:
            from pycocoevalcap.meteor.meteor import Meteor
            scorers.append(("METEOR", Meteor()))
        except (FileNotFoundError, OSError) as exc:
            scorer_notes.append(f"METEOR unavailable: {exc}")
        scorer = _CaptionScorer(scorers)
    metrics = score_records(records, predictions, caption_scorer=scorer)
    elapsed = time.perf_counter() - started
    latency_sorted = sorted(latencies)
    run_metadata = {"model_revision": a.model_revision, "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
                    "ordered_id_prompt_sha256": hashlib.sha256("\n".join(x["id"]+"\t"+x["prompt"] for x in records).encode()).hexdigest(),
                    "max_new_tokens": a.max_new_tokens, "do_sample": False,
                    "max_image_pixels": a.max_image_pixels,
                    "latency_seconds": {"mean": sum(latencies) / len(latencies),
                                        "median": latency_sorted[len(latency_sorted)//2],
                                        "p95": latency_sorted[max(0, min(len(latency_sorted)-1, math.ceil(.95*len(latency_sorted))-1))],
                                        "throughput_examples_per_second": len(records) / max(sum(latencies), 1e-12)},
                    "elapsed_seconds": elapsed}
    metadata_sidecar = a.manifest.with_suffix(a.manifest.suffix + ".meta.json")
    if metadata_sidecar.exists():
        run_metadata["dataset_metadata"] = json.loads(metadata_sidecar.read_text())
    if torch.cuda.is_available():
        run_metadata["cuda_peak_memory_bytes"] = {"allocated": int(torch.cuda.max_memory_allocated()),
                                                  "reserved": int(torch.cuda.max_memory_reserved())}
    payload = {"model_id": a.model_id, "model": str(a.model), "adapter": str(a.adapter) if a.adapter else None,
               "manifest": str(a.manifest), "run_metadata": run_metadata,
               "scorer_notes": scorer_notes,
               "n": len(records), "metrics": metrics,
               "records": [{**row, "prediction": pred, "latency_seconds": latency}
                           for row, pred, latency in zip(records, predictions, latencies)]}
    atomic_json_write(a.output, payload)
    partial_path.unlink(missing_ok=True)
    print(json.dumps({"model_id": a.model_id, "n": len(records), "metrics": metrics}, indent=2, ensure_ascii=False))


class _CaptionScorer:
    def __init__(self, scorers): self.scorers = scorers
    def compute_score(self, refs, hyps):
        values, names = [], []
        for name, scorer in self.scorers:
            score, _ = scorer.compute_score(refs, hyps)
            names.append(name); values.append(score)
        flat_names, flat_values = [], []
        for name, value in zip(names, values):
            if isinstance(value, (list, tuple)):
                flat_names.extend([f"{name}_{i+1}" for i in range(len(value))]); flat_values.extend(value)
            else:
                flat_names.append(name); flat_values.append(value)
        return dict(zip(flat_names, flat_values)), None


if __name__ == "__main__": main()
