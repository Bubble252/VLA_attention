"""Choose one deterministic visual phrase per Flickr30k Entities validation image."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities-val", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 means all validation images")
    args = parser.parse_args()
    by_image = defaultdict(list)
    for line in args.entities_val.read_text().splitlines():
        row = json.loads(line)
        by_image[row["image_id"]].append(row)
    chosen = []
    for image_id in sorted(by_image):
        # Fixed lexical selection prevents map quality from influencing sample choice.
        row = min(by_image[image_id], key=lambda value: (value["caption_index"] if "caption_index" in value else 0, value["phrase"].lower(), value["sample_id"]))
        chosen.append(row)
    if args.limit:
        chosen = chosen[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chosen))
    print(f"wrote {len(chosen)} deterministic image-level calibration rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
