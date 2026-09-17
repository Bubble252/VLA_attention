"""Run the Qwen2.5-VL phrase-score attribution interface audit (P1)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def find_last_subsequence(sequence: list[int], subsequence: list[int]) -> int:
    for index in range(len(sequence) - len(subsequence), -1, -1):
        if sequence[index : index + len(subsequence)] == subsequence:
            return index
    raise ValueError("teacher-forced phrase token span was not found in input_ids")


def main() -> int:
    import numpy as np
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False

    args.output.mkdir(parents=True, exist_ok=True)
    maps_dir = args.output / "maps"
    maps_dir.mkdir(exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True
    ).eval().cuda()
    merger = model.model.visual.merger
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        output.retain_grad()
        captured[:] = [output]

    handle = merger.register_forward_hook(hook)
    rows = []
    try:
        for raw in args.manifest.read_text().splitlines():
            sample = json.loads(raw)
            phrase = sample["phrase"]
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": str(args.dataset_root / sample["image_path"])},
                    {"type": "text", "text": "Answer using exactly this phrase:"},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": phrase}]},
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            images, videos = process_vision_info(messages)
            batch = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt")
            batch = {key: value.cuda() if hasattr(value, "cuda") else value for key, value in batch.items()}
            input_ids = batch["input_ids"][0].tolist()
            phrase_ids = processor.tokenizer(phrase, add_special_tokens=False)["input_ids"]
            phrase_start = find_last_subsequence(input_ids, phrase_ids)

            maps = []
            for _ in range(2):
                model.zero_grad(set_to_none=True)
                captured.clear()
                output = model(**batch, return_dict=True)
                log_probs = output.logits[0, phrase_start - 1 : phrase_start - 1 + len(phrase_ids)].float().log_softmax(-1)
                target = log_probs.gather(1, torch.tensor(phrase_ids, device=log_probs.device).unsqueeze(1)).sum()
                target.backward()
                features = captured[0]
                attribution = (features.grad.float() * features.float()).sum(-1).abs().detach().cpu().numpy()
                maps.append(attribution / max(float(attribution.sum()), 1e-12))
            grid_t, grid_h, grid_w = batch["image_grid_thw"][0].tolist()
            merge = model.config.vision_config.spatial_merge_size
            expected = grid_t * (grid_h // merge) * (grid_w // merge)
            if grid_t != 1 or len(maps[0]) != expected:
                raise ValueError(f"unexpected Qwen visual geometry: grid={grid_t, grid_h, grid_w}, tokens={len(maps[0])}")
            map_path = maps_dir / f"{sample['sample_id'].replace(':', '_')}.npy"
            np.save(map_path, maps[0].reshape(grid_h // merge, grid_w // merge))
            cosine = float(np.dot(maps[0], maps[1]) / (np.linalg.norm(maps[0]) * np.linalg.norm(maps[1]) + 1e-12))
            cosine = min(1.0, max(0.0, cosine))  # float32 roundoff can exceed 1 by epsilon
            visual_positions = (batch["input_ids"][0] == model.config.image_token_id).nonzero().flatten().tolist()
            rows.append({
                "sample_id": sample["sample_id"], "model_id": "qwen2.5-vl-7b",
                "target": "phrase_score", "map_path": str(map_path), "seed": args.seed,
                "scalar_definition": "teacher-forced assistant phrase token log-probability sum",
                "gradient_finite": bool(np.isfinite(maps[0]).all()), "repeatability": cosine,
                "grid": {"height": grid_h // merge, "width": grid_w // merge, "frame": "patch_grid",
                         "image_height": grid_h * model.config.vision_config.spatial_patch_size,
                         "image_width": grid_w * model.config.vision_config.spatial_patch_size,
                         "token_indices": visual_positions, "view_id": "image_0",
                         "metadata": {"grid_thw": f"{grid_t},{grid_h},{grid_w}", "spatial_merge_size": str(merge), "processor_use_fast": "false"}},
            })
    finally:
        handle.remove()
    report = {"capability": {"model_id": "qwen2.5-vl-7b", "model_revision": args.model.name,
              "supports_visual_tokens": True, "supports_hidden_states": True,
              "supported_targets": ["phrase_score"],
              "coordinate_notes": ["post-spatial-merge merger output", "processor use_fast=False"], "limitations": ["P1 uses one still image per input"]},
              "measurements": rows}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {len(rows)} measurements to {args.output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
