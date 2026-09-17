"""Build the caption-only V1--V4 SFT manifest from frozen Flickr train IDs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROMPT = "Describe the image in a single sentence as a caption."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities-train", type=Path, required=True)
    parser.add_argument("--captions-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    image_paths: dict[str, str] = {}
    for line in args.entities_train.read_text().splitlines():
        row = json.loads(line)
        image_paths.setdefault(row["image_id"], row["image_path"])
    rows = []
    with args.captions_csv.open(newline="") as handle:
        for item in csv.DictReader(handle):
            image_id = Path(item["filename"]).stem
            if image_id not in image_paths:
                continue
            for caption_index, caption in enumerate(json.loads(item["raw"])):
                caption = " ".join(caption.split())
                if caption:
                    rows.append({"sample_id": f"{image_id}:caption:{caption_index}", "image_id": image_id, "image_path": image_paths[image_id], "prompt": PROMPT, "caption": caption, "caption_index": caption_index, "split": "train"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(f"wrote {len(rows)} caption-SFT samples from {len(image_paths)} official train images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
