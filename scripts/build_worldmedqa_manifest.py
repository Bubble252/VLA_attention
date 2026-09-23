"""Build an image-backed WorldMedQA-V manifest from the dataset's TSV release."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import re
import sys
from pathlib import Path


def stable_key(value: str, seed: int) -> bytes:
    return hashlib.sha256((str(seed) + "\0" + value).encode()).digest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0, help="0 keeps the full official TSV collection")
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--revision", default="7d4b008bdba5c961a249cf44c9a552675d351ca5")
    a = p.parse_args()
    import requests

    files = [f"{country}_{language}_processed.tsv"
             for country in ("brazil", "israel", "japan", "spain")
             for language in ("english", "local")]
    rows = []
    csv.field_size_limit(sys.maxsize)
    for name in files:
        url = f"https://hf-mirror.com/datasets/WorldMedQA/V/resolve/{a.revision}/{name}"
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        text = response.content.decode("utf-8-sig")
        for record in csv.DictReader(text.splitlines(), delimiter="\t"):
            record["_source_file"] = name
            rows.append(record)
    ids = []
    duplicates = {}
    for row in rows:
        country = row["_source_file"].split("_")[0]
        language = "en" if "_english_" in row["_source_file"] else {
            "brazil": "pt", "israel": "he", "japan": "ja", "spain": "es"}[country]
        base_id = f"{row['_source_file']}:{row['index']}"
        occurrence = duplicates.get(base_id, 0)
        duplicates[base_id] = occurrence + 1
        sample_id = base_id if occurrence == 0 else f"{base_id}::dup{occurrence}"
        ids.append((sample_id, row, country, language))
    if a.limit:
        ids = sorted(ids, key=lambda x: stable_key(x[0], a.seed))[:a.limit]
    image_root = a.dataset_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    temp = a.output.with_name(a.output.name + ".tmp")
    seen = set()
    with temp.open("w", encoding="utf-8") as out:
        for sample_id, row, country, language in ids:
            encoded = re.sub(r"\s+", "", row["image"])
            payload = base64.b64decode(encoded, validate=True)
            suffix = ".png" if payload.startswith(b"\x89PNG\r\n\x1a\n") else ".jpg"
            image_name = hashlib.sha256(sample_id.encode()).hexdigest()[:20] + suffix
            image_path = image_root / image_name
            if not image_path.exists():
                image_path.write_bytes(payload)
            choices = [row[k].strip() for k in ("A", "B", "C", "D")]
            label = row.get("correct_option", row.get("answer", "")).strip().upper()
            if label not in ("A", "B", "C", "D") or not all(choices):
                raise ValueError(f"bad choice row: {sample_id} label={label!r}")
            prompt = "Choose the correct answer.\n" + row["question"].strip() + "\n" + "\n".join(
                f"{letter}. {choice}" for letter, choice in zip("ABCD", choices)) + "\nReply with the option letter only."
            item = {"id": sample_id, "task": "worldmedqa", "image": "images/" + image_name,
                    "prompt": prompt, "choices": choices, "answer": label,
                    "language": language, "question_type": country,
                    "source_file": row["_source_file"], "official_index": row["index"]}
            if sample_id in seen:
                raise ValueError(f"duplicate WorldMedQA ID {sample_id}")
            seen.add(sample_id)
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
    temp.replace(a.output)
    print(json.dumps({"dataset": "WorldMedQA-V", "revision": a.revision,
                      "source_files": files, "n": len(seen), "manifest": str(a.output)}, indent=2))


if __name__ == "__main__":
    main()
