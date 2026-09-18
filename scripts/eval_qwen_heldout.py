"""Evaluate Qwen phrase attribution on an image-disjoint held-out manifest.

The evaluator deliberately uses the same teacher-forced phrase-score target as
P1 and the training runners.  It does not update weights; checkpoints are
loaded only to compare V0--V4 on the fixed 64-image split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def find_last_subsequence(sequence: list[int], subsequence: list[int]) -> int:
    for i in range(len(sequence) - len(subsequence), -1, -1):
        if sequence[i : i + len(subsequence)] == subsequence:
            return i
    raise ValueError("phrase token span was not found in teacher-forced input")


def box_iou_from_top_mass(grid, boxes, *, image_width: int, image_height: int, fraction: float = 0.20) -> float:
    """IoU of GT boxes and the image-space box enclosing top-fraction cells."""
    import numpy as np
    flat = np.asarray(grid, dtype=np.float64).reshape(-1)
    k = max(1, int(np.ceil(flat.size * fraction)))
    ys, xs = np.unravel_index(np.argsort(flat)[-k:], grid.shape)
    left = float(xs.min()) * image_width / grid.shape[1]
    right = float(xs.max() + 1) * image_width / grid.shape[1]
    top = float(ys.min()) * image_height / grid.shape[0]
    bottom = float(ys.max() + 1) * image_height / grid.shape[0]
    pred = (left, top, right, bottom)
    best = 0.0
    for b in boxes:
        l, t, r, bb = map(float, b)
        il, it, ir, ib = max(left, l), max(top, t), min(right, r), min(bottom, bb)
        inter = max(0.0, ir - il) * max(0.0, ib - it)
        union = (right - left) * (bottom - top) + (r - l) * (bb - t) - inter
        best = max(best, inter / union if union > 0 else 0.0)
    return best


def main() -> int:
    import numpy as np
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from vla_attention.evaluation.spatial import mass_in_boxes, pointing_correct

    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--model-id", required=True)
    p.add_argument("--seed", type=int, default=23)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    proc = AutoProcessor.from_pretrained(a.model, local_files_only=True, use_fast=False)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation="eager"
    ).cuda().eval()
    model = base
    if a.checkpoint is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, a.checkpoint, is_trainable=False).eval()
    visual = model.base_model.model.model.visual if hasattr(model, "base_model") else model.model.visual
    # PEFT inference freezes the base model. Re-enable only the visual path so
    # the phrase-score backward pass has a valid feature gradient; no optimizer
    # is created and no weights are changed.
    for parameter in visual.parameters():
        parameter.requires_grad_(True)
    captured = []

    def hook(_module, _inputs, output):
        output.retain_grad()
        captured[:] = [output]

    handle = visual.merger.register_forward_hook(hook)
    rows = []
    maps_dir = a.output / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = [json.loads(x) for x in a.manifest.read_text().splitlines() if x.strip()]
    try:
        for idx, sample in enumerate(manifest_rows):
            image_path = a.dataset_root / sample["image_path"]
            phrase = sample["phrase"]
            caption = sample.get("caption", phrase)
            prompt = sample.get("prompt", "Describe the image in one sentence.")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": str(image_path)}, {"type": "text", "text": prompt}
            ]}, {"role": "assistant", "content": [{"type": "text", "text": caption}]}]
            text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            images, videos = process_vision_info(messages)
            batch = proc(text=[text], images=images, videos=videos, padding=True, return_tensors="pt")
            batch = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in batch.items()}
            ids = batch["input_ids"][0].tolist()
            phrase_ids = proc.tokenizer(phrase, add_special_tokens=False)["input_ids"]
            start = find_last_subsequence(ids, phrase_ids)
            model.zero_grad(set_to_none=True)
            captured.clear()
            out = model(**batch, return_dict=True)
            logp = out.logits[0, start - 1 : start - 1 + len(phrase_ids)].float().log_softmax(-1)
            target = logp.gather(1, torch.tensor(phrase_ids, device=logp.device).unsqueeze(1)).sum()
            target.backward()
            features = captured[0]
            attribution = (features.grad.float() * features.float()).sum(-1).abs().detach().cpu().numpy()
            grid_t, grid_h, grid_w = batch["image_grid_thw"][0].tolist()
            merge = base.config.vision_config.spatial_merge_size
            gh, gw = grid_h // merge, grid_w // merge
            grid = attribution.reshape(gh, gw)
            grid = grid / max(float(grid.sum()), 1e-12)
            with Image.open(image_path) as im:
                iw, ih = im.size
            boxes = sample["boxes_xyxy"]
            metrics = {
                "pointing": bool(pointing_correct(grid, boxes, image_width=iw, image_height=ih)),
                "mass_in_box": float(mass_in_boxes(grid, boxes, image_width=iw, image_height=ih)),
                "top20_box_iou": float(box_iou_from_top_mass(grid, boxes, image_width=iw, image_height=ih)),
            }
            map_path = maps_dir / f"{sample['sample_id'].replace(':', '_')}.npy"
            np.save(map_path, grid.astype(np.float32))
            rows.append({"sample_id": sample["sample_id"], "image_id": sample.get("image_id"), "phrase": phrase,
                         "map_path": str(map_path), "metrics": metrics, "gradient_finite": bool(np.isfinite(grid).all()),
                         "grid": [gh, gw], "image_size": [iw, ih]})
            print(f"[{idx + 1}/{len(manifest_rows)}] {sample['sample_id']} {metrics}", flush=True)
    finally:
        handle.remove()
    summary = {
        "model_id": a.model_id, "checkpoint": str(a.checkpoint) if a.checkpoint else None,
        "manifest": str(a.manifest), "n": len(rows),
        "pointing": float(np.mean([r["metrics"]["pointing"] for r in rows])),
        "mass_in_box": float(np.mean([r["metrics"]["mass_in_box"] for r in rows])),
        "top20_box_iou": float(np.mean([r["metrics"]["top20_box_iou"] for r in rows])),
    }
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "report.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
