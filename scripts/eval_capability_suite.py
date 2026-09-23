"""Generate and score normalized Lavender-style VLM capability manifests.

Input is JSONL with one record per image-question/task item. This keeps model
inference identical across benchmarks; official dataset adapters are responsible
for converting source data into the documented manifest fields.
"""
from __future__ import annotations

import argparse
import json
import re
import string
from collections import defaultdict
from pathlib import Path


def normalize_answer(value: str) -> str:
    text = value.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
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
    if not choices:
        match = re.search(r"\b([A-Z])\b", text.upper())
        return match.group(1) if match else ""
    match = re.search(r"(?:^|\b)([A-Z])(?:\b|[.):])", text.upper())
    if match and ord(match.group(1)) - ord("A") < len(choices):
        return match.group(1)
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
            metric = "vqa_consensus_approx" if task == "vqa" else "textvqa_consensus_approx"
            output[task] = {"count": len(values), metric: sum(values) / max(len(values), 1),
                            "metric_note": "approximate min(matching references/3,1); replace with official evaluator for final reporting"}
        elif task in ("pope", "binary"):
            labels = [row["label"].lower() for row, _ in pairs]
            pred_labels = [binary_label(pred) for _, pred in pairs]
            output[task] = {"count": len(labels), **binary_metrics(pred_labels, labels)}
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
            for row, pred in pairs:
                pair_correct[str(row["pair_id"])].append(binary_label(pred) == row["label"].lower())
            output[task] = {"count": len(pairs), "pair_count": len(pair_correct),
                            "pair_accuracy": sum(all(v) for v in pair_correct.values()) / max(len(pair_correct), 1)}
        else:
            raise ValueError(f"unsupported task={task!r}")
    return output


def load_model(model_path: Path, adapter: Path | None):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation="eager"
    ).cuda().eval()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False).eval()
    return model, processor


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
    a = p.parse_args()
    records = [json.loads(x) for x in a.manifest.read_text().splitlines() if x.strip()]
    if a.limit is not None:
        records = records[:a.limit]
    if not records:
        raise ValueError("manifest contains no records after applying --limit")
    for idx, row in enumerate(records):
        required = {"id", "task", "prompt"} - set(row)
        if required:
            raise ValueError(f"manifest row {idx} missing {sorted(required)}")
        if row.get("image") and not (a.dataset_root / row["image"]).is_file():
            raise FileNotFoundError(a.dataset_root / row["image"])
    model, processor = load_model(a.model, a.adapter)
    predictions = []
    for idx, row in enumerate(records):
        predictions.append(generate(model, processor, a.dataset_root, row, max_new_tokens=a.max_new_tokens))
        print(f"[{idx + 1}/{len(records)}] {row['id']}", flush=True)
    scorer = None
    if a.caption_metrics and any(r["task"] == "caption" for r in records):
        # Package's public API expects COCO objects; use its standard scorers directly.
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        scorer = _CaptionScorer([("Bleu", Bleu(4)), ("METEOR", Meteor()), ("ROUGE_L", Rouge()), ("CIDEr", Cider())])
    metrics = score_records(records, predictions, caption_scorer=scorer)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_id": a.model_id, "model": str(a.model), "adapter": str(a.adapter) if a.adapter else None,
               "manifest": str(a.manifest), "n": len(records), "metrics": metrics,
               "records": [{**row, "prediction": pred} for row, pred in zip(records, predictions)]}
    a.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
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
