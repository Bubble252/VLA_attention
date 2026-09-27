"""Single-process SD1.5 null-text phrase-map cache driver.

Unlike ``run_sd_cache.py`` this loads the diffusion pipeline once and reuses it
for every manifest row. It is intended for the 1k calibration and 10k train
cache, where reloading a 45 GB checkpoint per image is prohibitively slow.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    import numpy as np
    import torch
    from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionPipeline
    from PIL import Image
    from scripts.run_sd_ddim_map import find_subsequence
    from vla_attention.teachers.null_text import invert_ddim, optimize_null_text
    from vla_attention.teachers.sd_attention import install_cross_attention_capture

    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--inner-steps", type=int, default=10)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--resolution", type=int, default=16)
    p.add_argument("--seed", type=int, default=23)
    args = p.parse_args()

    args.cache.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    failures: list[dict[str, object]] = []
    progress = args.cache / "progress.json"

    torch.manual_seed(args.seed)
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        local_files_only=True,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("cuda")
    forward = DDIMScheduler.from_config(pipe.scheduler.config)
    inverse = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    store = install_cross_attention_capture(pipe.unet, conditional_cfg_half=True)
    empty = pipe.tokenizer(
        "", padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt"
    ).input_ids.cuda()

    def write_progress(done: int) -> None:
        progress.write_text(json.dumps({
            "total": len(rows), "completed": done, "failed": len(failures),
            "cache": str(args.cache), "seed": args.seed, "steps": args.steps,
            "inner_steps": args.inner_steps,
        }, indent=2) + "\n")

    completed = 0
    write_progress(0)
    for row in rows:
        key = row["sample_id"].replace(":", "_")
        metadata_path = args.cache / f"{key}.json"
        map_path = args.cache / f"{key}.npy"
        if metadata_path.exists() and map_path.exists():
            completed += 1
            write_progress(completed)
            continue
        try:
            torch.manual_seed(args.seed)
            store.maps.clear()
            image = Image.open(args.dataset_root / row["image_path"]).convert("RGB").resize((512, 512))
            pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).float()
            pixels = pixels.div(127.5).sub(1).to("cuda", dtype=torch.float32)
            with torch.no_grad():
                latent = pipe.vae.encode(pixels).latent_dist.mean * pipe.vae.config.scaling_factor
            tokens = pipe.tokenizer(
                row["caption"], padding="max_length", max_length=pipe.tokenizer.model_max_length,
                truncation=True, return_tensors="pt"
            )
            phrase_ids = pipe.tokenizer(row["phrase"], add_special_tokens=False)["input_ids"]
            phrase_positions = find_subsequence(tokens.input_ids[0].tolist(), phrase_ids)
            with torch.no_grad():
                conditional = pipe.text_encoder(tokens.input_ids.cuda())[0]
                unconditional = pipe.text_encoder(empty)[0]
            trajectory = invert_ddim(pipe.unet, inverse, latent, conditional, steps=args.steps, device="cuda")
            optimized = optimize_null_text(
                pipe.unet, forward, trajectory, unconditional, conditional,
                steps=args.steps, guidance_scale=args.guidance_scale,
                inner_steps=args.inner_steps, device="cuda",
            )
            current = trajectory[-1]
            forward.set_timesteps(args.steps, device="cuda")
            with torch.no_grad():
                for timestep, uncond in zip(forward.timesteps, optimized.unconditional_embeddings):
                    pair = torch.cat([current, current])
                    output = pipe.unet(
                        pair, timestep,
                        encoder_hidden_states=torch.cat([uncond, conditional]),
                    ).sample
                    uncond_noise, cond_noise = output.chunk(2)
                    current = forward.step(
                        uncond_noise + args.guidance_scale * (cond_noise - uncond_noise),
                        timestep, current,
                    ).prev_sample
            tensors = store.maps.get(args.resolution, [])
            if not tensors:
                raise ValueError(f"no attention tensors at resolution {args.resolution}")
            attention = torch.cat(tensors, dim=0).mean(dim=0)
            spatial = attention[:, phrase_positions].mean(dim=-1).reshape(args.resolution, args.resolution).numpy()
            spatial /= max(float(spatial.sum()), 1e-12)
            np.save(map_path, spatial)
            metadata_path.write_text(json.dumps({
                "method": "ddim_nulltext_fp32_v2", "formal_lavender_exact": True,
                "sample_id": row["sample_id"], "caption": row["caption"], "phrase": row["phrase"],
                "phrase_token_positions": phrase_positions, "inversion_steps": args.steps,
                "inner_steps": args.inner_steps, "guidance_scale": args.guidance_scale,
                "attention_resolution": args.resolution, "seed": args.seed,
                "attention_tensors": len(tensors),
                "mean_reconstruction_mse": sum(optimized.reconstruction_losses) / len(optimized.reconstruction_losses),
                "map_path": str(map_path),
            }, indent=2) + "\n")
            completed += 1
        except Exception as exc:  # keep the cache job progressing and audit failures
            failures.append({"sample_id": row["sample_id"], "error": repr(exc)})
        finally:
            store.maps.clear()
            write_progress(completed)
    (args.cache / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    print(json.dumps({"total": len(rows), "completed": completed, "failures": len(failures)}))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
