"""Build reproducible Flickr30k Entities manifests after archive verification.

The script does not download data and does not assume that ZIP extraction keeps a
particular top-level directory.  It emits only metadata; images and annotation
archives remain in VEPFS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


PHRASE = re.compile(r"\[/EN#(?P<chain>\d+)(?:/[^\s\]]+)*\s+(?P<phrase>[^\]]+)\]")


def locate(root: Path, relative_name: str) -> Path:
    matches = sorted(root.rglob(relative_name))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {relative_name} under {root}, found {matches}")
    return matches[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def boxes_by_chain(xml_path: Path) -> dict[str, list[list[int]]]:
    root = ET.parse(xml_path).getroot()
    boxes: dict[str, list[list[int]]] = defaultdict(list)
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        coords = [int(bndbox.findtext(key, "0")) for key in ("xmin", "ymin", "xmax", "ymax")]
        if coords[2] <= coords[0] or coords[3] <= coords[1]:
            continue
        for name in obj.findall("name"):
            if name.text:
                boxes[name.text].append(coords)
    return boxes


def phrase_records(sentence_path: Path, chain_boxes: dict[str, list[list[int]]]) -> list[tuple[int, int, str, str, str, list[list[int]]]]:
    records = []
    for caption_index, line in enumerate(sentence_path.read_text(errors="replace").splitlines()):
        caption = " ".join(PHRASE.sub(lambda match: match.group("phrase"), line).split())
        for phrase_index, match in enumerate(PHRASE.finditer(line)):
            chain = match.group("chain")
            phrase = " ".join(match.group("phrase").split())
            if chain != "0" and phrase and chain_boxes.get(chain):
                records.append((caption_index, phrase_index, chain, phrase, caption, chain_boxes[chain]))
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--p1-images", type=int, default=20)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    sentences = locate(root, "Sentences")
    annotations = locate(root, "Annotations")
    splits = {name: locate(root, f"{name}.txt") for name in ("train", "val", "test")}
    image_index = {path.stem: path for path in root.rglob("*.jpg")}
    if not image_index:
        raise ValueError("no JPEG image files found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"seed": args.seed, "splits": {}, "image_root_count": len(image_index)}
    all_split_records: dict[str, list[dict[str, object]]] = {}
    for split, split_file in splits.items():
        image_ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        records: list[dict[str, object]] = []
        for image_id in image_ids:
            sentence_path, xml_path, image_path = sentences / f"{image_id}.txt", annotations / f"{image_id}.xml", image_index.get(image_id)
            if not (sentence_path.is_file() and xml_path.is_file() and image_path):
                continue
            for caption_index, phrase_index, chain, phrase, caption, boxes in phrase_records(sentence_path, boxes_by_chain(xml_path)):
                records.append({
                    "sample_id": f"{image_id}:c{caption_index}:p{phrase_index}:e{chain}",
                    "image_id": image_id,
                    "image_path": str(image_path.relative_to(root)),
                    "phrase": phrase,
                    "caption": caption,
                    "entity_id": chain,
                    "boxes_xyxy": boxes,
                    "split": split,
                })
        output = args.output_dir / f"flickr30k_entities_{split}.jsonl"
        output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
        summary["splits"][split] = {"images_listed": len(image_ids), "phrase_box_records": len(records), "manifest": output.name}
        all_split_records[split] = records

    by_image: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in all_split_records["val"]:
        by_image[row["image_id"]].append(row)
    selected_ids = sorted(by_image)
    if len(selected_ids) < args.p1_images:
        raise ValueError(f"only {len(selected_ids)} calibration images have box-linked phrases")
    selected_ids = sorted(random.Random(args.seed).sample(selected_ids, args.p1_images))
    p1_rows = []
    for image_id in selected_ids:
        row = sorted(by_image[image_id], key=lambda value: value["sample_id"])[0].copy()
        row["image_sha256"] = sha256(root / row["image_path"])
        p1_rows.append(row)
    p1_path = args.output_dir / "p1_qwen25_calibration_20.jsonl"
    p1_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in p1_rows))
    summary["p1"] = {"source_split": "val", "images": len(p1_rows), "manifest": p1_path.name}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
