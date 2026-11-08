#!/usr/bin/env python3
"""Cache frozen Qwen visual/language features for Goal18 eval correction.

The Goal18 eval sidecar has masks and images but no point labels.  C2 only
needs the frozen visual/language summaries, so this extractor deliberately
does not read or require object/destination labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from starVLA.model.modules.vlm.QWen2_5 import IMAGE_TOKEN_INDEX, _QWen_VL_Interface
from c1_point_utils import assign_visual_views, image_token_counts, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = read_jsonl(args.sidecar)
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("no eval rows selected")

    config = OmegaConf.load(args.config)
    interface = _QWen_VL_Interface(config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    interface.model.to(device)
    interface.model.eval()
    merge_size = int(interface.model.config.vision_config.spatial_merge_size)
    feature_dir = args.output / "features"
    feature_dir.mkdir(parents=True, exist_ok=False)
    index_path = args.output / "index.jsonl"
    visual_counts: list[int] = []
    language_counts: list[int] = []

    with index_path.open("w") as handle:
        for row_index, row in enumerate(rows):
            image = Image.open(row["image_path"]).convert("RGB")
            wrist = Image.open(row["wrist_image_path"]).convert("RGB")
            inputs = interface.build_qwenvl_inputs(
                images=[[image, wrist]],
                instructions=[str(row["language"])],
            )
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = interface(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
            hidden = outputs.hidden_states[-1][0]
            input_ids = inputs["input_ids"][0]
            attention_mask = inputs.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            else:
                attention_mask = attention_mask[0].bool()
            image_mask = input_ids == IMAGE_TOKEN_INDEX
            visual_positions = image_mask.nonzero(as_tuple=False).flatten()
            if visual_positions.numel() == 0:
                raise ValueError(f"{row['row_id']}: no visual tokens")
            grid = inputs.get("image_grid_thw")
            if grid is None:
                raise ValueError(f"{row['row_id']}: no image_grid_thw")
            counts = image_token_counts(grid, merge_size)
            view_ids, mapping = assign_visual_views(visual_positions, counts)
            language_positions = (attention_mask & ~image_mask).nonzero(as_tuple=False).flatten()
            if language_positions.numel() == 0:
                raise ValueError(f"{row['row_id']}: no language tokens")

            visual_tokens = hidden[visual_positions].detach().to(torch.float16).cpu()
            language_tokens = hidden[language_positions].detach().to(torch.float16).cpu()
            feature_path = feature_dir / f"{row['row_id']}.npz"
            np.savez_compressed(
                feature_path,
                visual_tokens=visual_tokens.numpy(),
                visual_view_ids=view_ids.numpy(),
                language_tokens=language_tokens.numpy(),
            )
            handle.write(
                json.dumps(
                    {
                        "row_id": row["row_id"],
                        "shared_init_id": row.get("shared_init_id"),
                        "prompt_id": row.get("prompt_id"),
                        "language": row.get("language"),
                        "feature_path": str(feature_path),
                        "image_path": row.get("image_path"),
                        "wrist_image_path": row.get("wrist_image_path"),
                        "mask_path": row.get("mask_path"),
                        "image_size": int(row.get("policy_image_size", 224)),
                        "visual_token_count": int(visual_tokens.shape[0]),
                        "language_token_count": int(language_tokens.shape[0]),
                        "view_mapping": mapping,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            visual_counts.append(int(visual_tokens.shape[0]))
            language_counts.append(int(language_tokens.shape[0]))
            if (row_index + 1) % 10 == 0:
                print(
                    json.dumps({"processed": row_index + 1, "total": len(rows)}),
                    flush=True,
                )

    summary: dict[str, Any] = {
        "schema": "goal18-eval-features-v1",
        "status": "complete",
        "config": str(args.config),
        "sidecar": str(args.sidecar),
        "rows": len(rows),
        "visual_token_count": {
            "min": min(visual_counts),
            "max": max(visual_counts),
            "mean": float(np.mean(visual_counts)),
        },
        "language_token_count": {
            "min": min(language_counts),
            "max": max(language_counts),
            "mean": float(np.mean(language_counts)),
        },
        "notes": [
            "Qwen is frozen; no weights are updated.",
            "Only deployable visual/language hidden states are cached.",
            "No object/destination labels are read by this extractor.",
        ],
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
