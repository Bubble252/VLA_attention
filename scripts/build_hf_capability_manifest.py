"""Build deterministic, image-backed capability JSONL from public HF datasets.

This is for fixed pilot manifests. It never uses model predictions or selects
items by difficulty; examples are chosen by hashing official sample IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def stable_score(sample_id: str, seed: int) -> bytes:
    return hashlib.sha256((str(seed) + "\0" + sample_id).encode()).digest()


def field_id(row: dict, task: str, index: int) -> str:
    if task == "mme" and row.get("question_id") is not None:
        return f"{row['question_id']}::{index}"
    for key in ("question_id", "cocoid", "id", "image_id"):
        if row.get(key) is not None:
            return str(row[key])
    return f"row-{index:08d}"


def normalize_row(row: dict, task: str, sample_id: str, image_rel: str) -> dict:
    if task == "caption":
        refs = row.get("sentences_raw") or row.get("captions") or row.get("references")
        return {"id": sample_id, "task": "caption", "image": image_rel,
                "prompt": "Describe the image in one sentence.", "references": list(refs)}
    if task in ("vqa", "textvqa"):
        question = row["question"]
        answers = row.get("answers") or row.get("answers_original")
        if answers and isinstance(answers[0], dict):
            answers = [x.get("answer", "") for x in answers]
        return {"id": sample_id, "task": task, "image": image_rel,
                "prompt": f"Answer the question with a short answer. Question: {question}",
                "answers": [str(x) for x in answers if str(x).strip()],
                "question_type": row.get("question_type"), "answer_type": row.get("answer_type")}
    if task in ("pope", "binary"):
        answer = str(row.get("answer", row.get("label", ""))).strip().lower()
        label = "yes" if answer.startswith("yes") else "no" if answer.startswith("no") else answer
        subset = row.get("category", row.get("split", row.get("subset", "unspecified")))
        return {"id": sample_id, "task": "pope", "image": image_rel,
                "prompt": "Answer yes or no only. " + str(row["question"]),
                "label": label, "category": str(subset), "question_id": str(row.get("question_id", sample_id))}
    if task == "mme":
        answer = str(row.get("answer", "")).strip().lower()
        label = "yes" if answer in ("yes", "true") else "no" if answer in ("no", "false") else answer
        return {"id": sample_id, "task": "mme_pair", "image": image_rel,
                "prompt": "Answer yes or no only. " + str(row["question"]),
                "label": label, "category": str(row.get("category", "unspecified")),
                "pair_id": str(row.get("_pair_id") or (str(row.get("category", "unspecified")) + ":" + str(row.get("question_id", sample_id)))),
                "question_id": str(row.get("question_id", sample_id))}
    raise ValueError(f"unsupported source task {task}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--config")
    p.add_argument("--split", required=True)
    p.add_argument("--task", choices=["caption", "vqa", "textvqa", "pope", "mme"], required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, required=True)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--subset-field", help="optional field whose groups must be sampled evenly")
    p.add_argument("--category-label", help="override category metadata, e.g. POPE official subset name")
    a = p.parse_args()
    from datasets import load_dataset

    data = load_dataset(a.dataset, name=a.config, split=a.split)
    id_key = next((k for k in ("question_id", "cocoid", "id", "image_id") if k in data.column_names), None)
    if id_key is None:
        raise ValueError(f"dataset has no stable ID field: {data.column_names}")
    candidates = list(range(len(data)))
    if a.subset_field:
        groups = {}
        for i in candidates:
            groups.setdefault(str(data[i].get(a.subset_field, "unspecified")), []).append(i)
        chosen = []
        quota = max(1, a.limit // max(len(groups), 1))
        for group in sorted(groups):
            ordered = sorted(groups[group], key=lambda i: stable_score(str(data[i][id_key]), a.seed))
            chosen.extend(ordered[:quota])
        if len(chosen) < a.limit:
            left = [i for i in candidates if i not in set(chosen)]
            left.sort(key=lambda i: stable_score(str(data[i][id_key]), a.seed))
            chosen.extend(left[:a.limit-len(chosen)])
        indices = chosen[:a.limit]
    elif a.task == "mme":
        pairs = {}
        pair_occurrences = {}
        for i in candidates:
            row = data[i]
            image_key = str(row.get("category", "unspecified")) + ":" + str(row.get("question_id", row[id_key]))
            occurrence = pair_occurrences.get(image_key, 0)
            pair_occurrences[image_key] = occurrence + 1
            key = f"{image_key}:pair{occurrence // 2}"
            pairs.setdefault(key, []).append(i)
        ordered_pairs = sorted(pairs, key=lambda key: stable_score(key, a.seed))
        selected_pairs = ordered_pairs[:max(1, a.limit // 2)]
        indices = []
        for key in selected_pairs:
            members = pairs[key]
            if len(members) != 2:
                continue
            indices.extend(members)
        if len(indices) != a.limit:
            raise ValueError(f"expected {a.limit} MME rows in complete pairs; selected {len(indices)}")
    else:
        indices = sorted(candidates, key=lambda i: stable_score(str(data[i][id_key]), a.seed))[:a.limit]
    image_dir = a.dataset_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_name(a.output.name + ".tmp")
    seen = set()
    with tmp.open("w", encoding="utf-8") as out:
        for index in indices:
            row = data[index]
            sample_id = field_id(row, a.task, index)
            if sample_id in seen:
                raise ValueError(f"duplicate stable sample ID: {sample_id}")
            seen.add(sample_id)
            image = row.get("image")
            if image is None:
                raise ValueError(f"row {sample_id} has no embedded image")
            image_name = hashlib.sha256(sample_id.encode()).hexdigest()[:20] + ".jpg"
            image_path = image_dir / image_name
            if not image_path.exists():
                image.convert("RGB").save(image_path, format="JPEG", quality=95)
            if a.task == "mme":
                # MME orders each image's positive/negative questions in adjacent pairs.
                prior = sum(1 for j in indices[:indices.index(index)]
                            if str(data[j].get("category", "unspecified")) == str(row.get("category", "unspecified"))
                            and str(data[j].get("question_id", data[j].get(id_key))) == str(row.get("question_id", row.get(id_key))))
                row = dict(row)
                row["_pair_id"] = f"{row.get('category', 'unspecified')}:{row.get('question_id', row.get(id_key))}:pair{prior // 2}"
            record = normalize_row(row, a.task, sample_id, "images/" + image_name)
            if a.category_label:
                record["category"] = a.category_label
                if a.task == "pope":
                    record["id"] = a.category_label + ":" + record["id"]
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            if len(seen) % 50 == 0:
                print(f"written={len(seen)}/{len(indices)}", flush=True)
    tmp.replace(a.output)
    print(json.dumps({"dataset": a.dataset, "split": a.split, "task": a.task,
                      "n": len(seen), "manifest": str(a.output)}, indent=2))


if __name__ == "__main__":
    main()
