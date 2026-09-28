"""Attach one deterministic Flickr30k Entities phrase to each caption-SFT row.

The caption, image and caption index remain unchanged; the phrase/box fields
only provide the semantic teacher target for V3/V4. Rows without an aligned
phrase are reported and excluded so every semantic group uses one immutable
sample intersection.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def caption_index(sample_id: str) -> int:
    if ":caption:" in sample_id:
        return int(sample_id.rsplit(":caption:", 1)[1])
    parts = sample_id.split(":")
    for part in parts[1:]:
        if part.startswith("c") and part[1:].isdigit():
            return int(part[1:])
    raise ValueError(f"cannot parse caption index from {sample_id}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--caption-manifest", type=Path, required=True)
    p.add_argument("--entities-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--failures", type=Path, required=True)
    args = p.parse_args()
    by_key: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for line in args.entities_manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        by_key[(str(row["image_id"]), caption_index(str(row["sample_id"])))].append(row)
    output, failures = [], []
    for line in args.caption_manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["image_id"]), int(row["caption_index"]))
        candidates = by_key.get(key, [])
        if not candidates:
            failures.append({"sample_id": row["sample_id"], "reason": "no_entity_phrase_for_caption"})
            continue
        # Stable lexical phrase/entity choice, independent of teacher output.
        chosen = min(candidates, key=lambda x: (str(x.get("phrase", "")).casefold(), str(x.get("entity_id", "")), str(x["sample_id"])))
        merged = dict(row)
        for field in ("phrase", "entity_id", "boxes_xyxy"):
            merged[field] = chosen[field]
        merged["alignment_source_sample_id"] = chosen["sample_id"]
        output.append(merged)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.failures.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in output))
    args.failures.write_text(json.dumps(failures, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"input": len(output) + len(failures), "output": len(output), "failures": len(failures)}))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
