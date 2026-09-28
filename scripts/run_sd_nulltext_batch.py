"""Single-process SD1.5 null-text phrase-map cache driver.

Unlike ``run_sd_cache.py`` this loads the diffusion pipeline once and reuses it
for every manifest row. It is intended for the 1k calibration and 10k train
cache, where reloading a 45 GB checkpoint per image is prohibitively slow.
"""
from __future__ import annotations

import argparse
import json
import os
import hashlib
import time
from pathlib import Path


def main() -> int:
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
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

    def hash_file(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    manifest_sha = hash_file(args.manifest)
    code_sha = hash_file(__file__)
    def atomic_json(path, payload):
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)

    args.cache.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    for row in rows:
        if not all(row.get(k) for k in ('sample_id', 'image_path', 'caption', 'phrase')):
            raise ValueError('manifest missing sample/image/caption/phrase; refuse GPU load')
        key = row['sample_id'].replace(':', '_')
        existing = args.cache / f'{key}.json'
        if existing.exists():
            meta = json.loads(existing.read_text())
            if (meta.get('method') != 'ddim_nulltext_fp32_reconstruction_v3'
                    or meta.get('seed') != args.seed
                    or meta.get('inversion_steps') != args.steps
                    or meta.get('inner_steps') != args.inner_steps
                    or meta.get('guidance_scale') != args.guidance_scale
                    or meta.get('attention_resolution') != args.resolution
                    or meta.get('phrase_occurrence','first') != row.get('phrase_occurrence','first')
                    or meta.get('numerics') != 'fp32,deterministic_algorithms,math_sdpa,tf32_off,cublas4096:8'
                    or meta.get('caption') != row['caption'] or meta.get('phrase') != row['phrase']):
                raise ValueError(f'incompatible existing cache; use a new versioned directory: {existing}')
    failures: list[dict[str, object]] = []
    progress = args.cache / "progress.json"

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        local_files_only=True,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("cuda")
    for module in (pipe.unet, pipe.vae, pipe.text_encoder):
        module.eval().requires_grad_(False)
    forward = DDIMScheduler.from_config(pipe.scheduler.config)
    inverse = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    original_processors = dict(pipe.unet.attn_processors)
    empty = pipe.tokenizer(
        "", padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt"
    ).input_ids.cuda()

    def write_progress(done: int) -> None:
        atomic_json(progress, {
            "total": len(rows), "completed": done, "failed": len(failures),
            "cache": str(args.cache), "seed": args.seed, "steps": args.steps,
            "inner_steps": args.inner_steps, "updated_at_unix": time.time(),
            "manifest_sha256": manifest_sha, "extractor_sha256": code_sha,
        })
        atomic_json(args.cache/'failures.json',failures)

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
            if row.get('phrase_occurrence')=='last':
                sequence=tokens.input_ids[0].tolist()
                matches=[list(range(i,i+len(phrase_ids))) for i in range(len(sequence)-len(phrase_ids)+1)
                         if sequence[i:i+len(phrase_ids)]==phrase_ids]
                phrase_positions=matches[-1]
            with torch.no_grad():
                conditional = pipe.text_encoder(tokens.input_ids.cuda())[0]
                unconditional = pipe.text_encoder(empty)[0]
            trajectory = invert_ddim(pipe.unet, inverse, latent, conditional, steps=args.steps, device="cuda")
            optimized = optimize_null_text(
                pipe.unet, forward, trajectory, unconditional, conditional,
                steps=args.steps, guidance_scale=args.guidance_scale,
                inner_steps=args.inner_steps, device="cuda",
            )
            # Capture only the final, reconstruction-conditioned forward pass.
            # Installing this before inversion would mix inversion and null-text
            # optimization attention (1400 tensors) with the intended 20-step
            # reconstruction capture (100 tensors at the current SD1.5 layout).
            store = install_cross_attention_capture(pipe.unet, conditional_cfg_half=True)
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
            if args.resolution==16 and len(tensors)!=args.steps*5:
                raise ValueError('SD1.5 capture count differs from audited reconstruction-only layout')
            attention = torch.cat(tensors, dim=0).mean(dim=0)
            spatial = attention[:, phrase_positions].mean(dim=-1).reshape(args.resolution, args.resolution).numpy()
            if not np.isfinite(spatial).all() or (spatial < 0).any() or spatial.sum() <= 0:
                raise ValueError('invalid spatial attention map')
            if not np.isfinite(optimized.reconstruction_losses).all():
                raise ValueError('nonfinite null-text reconstruction loss')
            spatial /= max(float(spatial.sum()), 1e-12)
            temporary_map=map_path.with_suffix('.npy.tmp')
            with temporary_map.open('wb') as f: np.save(f, spatial)
            temporary_map.replace(map_path)
            atomic_json(metadata_path, {
                "method": "ddim_nulltext_fp32_reconstruction_v3", "formal_lavender_exact": False,
                "capture_stage": "final_reconstruction_only", "gpu_parity_validation": "pending",
                "sample_id": row["sample_id"], "caption": row["caption"], "phrase": row["phrase"],
                "phrase_token_positions": phrase_positions, "inversion_steps": args.steps,
                "phrase_occurrence": row.get('phrase_occurrence','first'),
                "inner_steps": args.inner_steps, "guidance_scale": args.guidance_scale,
                "attention_resolution": args.resolution, "seed": args.seed,
                "attention_tensors": len(tensors),
                "mean_reconstruction_mse": sum(optimized.reconstruction_losses) / len(optimized.reconstruction_losses),
                "map_path": str(map_path),
                "map_sha256": hash_file(map_path), "image_sha256":hash_file(args.dataset_root/row['image_path']),
                "instruction_sha256":hashlib.sha256(row.get('instruction',row['caption']).encode()).hexdigest(),
                "manifest_sha256":manifest_sha, "extractor_sha256":code_sha,
                "teacher_frozen":True,"image_transform":"RGB->PIL resize(512,512),stretch,no_crop",
                "numerics":"fp32,deterministic_algorithms,math_sdpa,tf32_off,cublas4096:8",
                "source_provenance":{k:row[k] for k in ('episode_id','timestep','camera','split','role','instruction') if k in row},
            })
            completed += 1
        except Exception as exc:  # keep the cache job progressing and audit failures
            failures.append({"sample_id": row["sample_id"], "error": repr(exc)})
            print(json.dumps(failures[-1]),flush=True)
            if len(failures)>=3 and completed==0:
                raise RuntimeError('First three samples failed; stop before wasting full batch') from exc
        finally:
            if "store" in locals():
                store.maps.clear()
            pipe.unet.set_attn_processor(dict(original_processors))
            write_progress(completed)
    (args.cache / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    print(json.dumps({"total": len(rows), "completed": completed, "failures": len(failures)}))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
