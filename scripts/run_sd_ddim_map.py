"""Extract a DDIM-inversion-conditioned SD1.5 phrase map for calibration studies.

The result is deliberately marked non-final because it does not optimize
null-text embeddings.  It validates the image-conditioned path and supplies a
controlled intermediate baseline for the later exact Lavender comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def find_subsequence(sequence: list[int], query: list[int]) -> list[int]:
    for start in range(len(sequence) - len(query) + 1):
        if sequence[start : start + len(query)] == query:
            return list(range(start, start + len(query)))
    raise ValueError("phrase tokens were not found in the full caption tokenization")


def main() -> int:
    import numpy as np
    import torch
    from diffusers import DDIMInverseScheduler, StableDiffusionPipeline
    from PIL import Image
    from vla_attention.teachers.ddim import invert_latent
    from vla_attention.teachers.sd_attention import install_cross_attention_capture

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--caption", required=True)
    parser.add_argument("--phrase", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16, local_files_only=True, safety_checker=None, requires_safety_checker=False).to("cuda")
    inverse = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    store = install_cross_attention_capture(pipe.unet, conditional_cfg_half=False)
    image = Image.open(args.image).convert("RGB").resize((512, 512))
    pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).float().div(127.5).sub(1).to("cuda", dtype=torch.float16)
    with torch.no_grad():
        latent = pipe.vae.encode(pixels).latent_dist.mean * pipe.vae.config.scaling_factor
    caption_tokens = pipe.tokenizer(args.caption, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    phrase_tokens = pipe.tokenizer(args.phrase, add_special_tokens=False)["input_ids"]
    phrase_positions = find_subsequence(caption_tokens.input_ids[0].tolist(), phrase_tokens)
    with torch.no_grad():
        embedding = pipe.text_encoder(caption_tokens.input_ids.cuda())[0]
        _ = invert_latent(pipe.unet, inverse, latent, embedding, steps=args.steps, device="cuda")
    if args.resolution not in store.maps:
        raise ValueError(f"no captured cross attention at requested resolution {args.resolution}; got {sorted(store.maps)}")
    tensors = store.maps[args.resolution]
    attention = torch.cat(tensors, dim=0).mean(dim=0)  # q, text-token
    spatial_map = attention[:, phrase_positions].mean(dim=-1).reshape(args.resolution, args.resolution).numpy()
    spatial_map /= max(float(spatial_map.sum()), 1e-12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_suffix(".npy"), spatial_map)
    metadata = {
        "method": "ddim_conditional_no_nulltext",
        "formal_lavender_exact": False,
        "formal_calibration_eligible": False,
        "caption": args.caption,
        "phrase": args.phrase,
        "phrase_token_positions": phrase_positions,
        "inversion_steps": args.steps,
        "attention_resolution": args.resolution,
        "seed": args.seed,
        "attention_tensors": len(tensors),
        "map_path": str(args.output.with_suffix(".npy")),
    }
    args.output.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
