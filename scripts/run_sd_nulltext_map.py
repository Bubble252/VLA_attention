"""Exact-style SD null-text inversion attention-map smoke/calibration runner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_sd_ddim_map import find_subsequence


def main() -> int:
    import numpy as np
    import torch
    from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionPipeline
    from PIL import Image
    from vla_attention.teachers.null_text import invert_ddim, optimize_null_text
    from vla_attention.teachers.sd_attention import install_cross_attention_capture

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--caption", required=True)
    parser.add_argument("--phrase", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--inner-steps", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16, local_files_only=True, safety_checker=None, requires_safety_checker=False).to("cuda")
    forward = DDIMScheduler.from_config(pipe.scheduler.config)
    inverse = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    image = Image.open(args.image).convert("RGB").resize((512, 512))
    pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).float().div(127.5).sub(1).to("cuda", dtype=torch.float16)
    with torch.no_grad():
        latent = pipe.vae.encode(pixels).latent_dist.mean * pipe.vae.config.scaling_factor
    tokens = pipe.tokenizer(args.caption, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    phrase_ids = pipe.tokenizer(args.phrase, add_special_tokens=False)["input_ids"]
    phrase_positions = find_subsequence(tokens.input_ids[0].tolist(), phrase_ids)
    with torch.no_grad():
        conditional = pipe.text_encoder(tokens.input_ids.cuda())[0]
        unconditional = pipe.text_encoder(pipe.tokenizer("", padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt").input_ids.cuda())[0]
    trajectory = invert_ddim(pipe.unet, inverse, latent, conditional, steps=args.steps, device="cuda")
    optimized = optimize_null_text(pipe.unet, forward, trajectory, unconditional, conditional, steps=args.steps, guidance_scale=args.guidance_scale, inner_steps=args.inner_steps, device="cuda")
    store = install_cross_attention_capture(pipe.unet, conditional_cfg_half=True)
    current = trajectory[-1]
    forward.set_timesteps(args.steps, device="cuda")
    with torch.no_grad():
        for timestep, uncond in zip(forward.timesteps, optimized.unconditional_embeddings):
            pair = torch.cat([current, current])
            output = pipe.unet(pair, timestep, encoder_hidden_states=torch.cat([uncond, conditional])).sample
            uncond_noise, cond_noise = output.chunk(2)
            current = forward.step(uncond_noise + args.guidance_scale * (cond_noise - uncond_noise), timestep, current).prev_sample
    tensors = store.maps.get(args.resolution, [])
    if not tensors:
        raise ValueError(f"no attention tensors at resolution {args.resolution}")
    attention = torch.cat(tensors, dim=0).mean(dim=0)
    spatial = attention[:, phrase_positions].mean(dim=-1).reshape(args.resolution, args.resolution).numpy()
    spatial /= max(float(spatial.sum()), 1e-12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_suffix(".npy"), spatial)
    metadata = {"method": "ddim_nulltext", "formal_lavender_exact": True, "caption": args.caption, "phrase": args.phrase, "phrase_token_positions": phrase_positions, "inversion_steps": args.steps, "inner_steps": args.inner_steps, "guidance_scale": args.guidance_scale, "attention_resolution": args.resolution, "seed": args.seed, "attention_tensors": len(tensors), "mean_reconstruction_mse": sum(optimized.reconstruction_losses) / len(optimized.reconstruction_losses), "map_path": str(args.output.with_suffix(".npy"))}
    args.output.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
