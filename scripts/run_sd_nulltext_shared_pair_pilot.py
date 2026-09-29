"""Run an isolated two-member semantic shared-computation pilot.

The input is produced by ``prepare_semantic_shared_pilot.py``. One complete
FP32 SD1.5 inversion/null-text/final-reconstruction pass is performed per
source/target pair, then two phrase maps are exported from the same captured
attention tensors. This script refuses formal cache paths and is intentionally
not used by the production queue.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


NUMERICS = "fp32,deterministic_algorithms,math_sdpa,tf32_off,cublas4096:8"

# Make the repository importable even when the script is launched by an
# absolute path from a scheduler or a remote shell.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_ROOT = _REPO_ROOT / "src"
if _SRC_ROOT.is_dir() and str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _phrase_positions(tokenizer: Any, caption: str, phrase: str, occurrence: str) -> list[int]:
    from scripts.run_sd_ddim_map import find_subsequence

    tokens = tokenizer(
        caption,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    phrase_ids = tokenizer(phrase, add_special_tokens=False)["input_ids"]
    sequence = tokens.input_ids[0].tolist()
    if occurrence == "last":
        matches = [
            list(range(i, i + len(phrase_ids)))
            for i in range(len(sequence) - len(phrase_ids) + 1)
            if sequence[i : i + len(phrase_ids)] == phrase_ids
        ]
        if not matches:
            raise ValueError(f"phrase not found: {phrase!r}")
        return matches[-1]
    positions = find_subsequence(sequence, phrase_ids)
    if not positions:
        raise ValueError(f"phrase not found: {phrase!r}")
    return positions


def main() -> int:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionPipeline
    from PIL import Image
    from vla_attention.teachers.null_text import invert_ddim, optimize_null_text
    from vla_attention.teachers.sd_attention import install_cross_attention_capture

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--inner-steps", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    cache_text = str(args.cache)
    if "pilot" not in cache_text.lower():
        raise ValueError("shared pilot refuses a cache path without 'pilot'")
    if "semantic_train_task0_v3" in cache_text or "teacher_maps/F1_" in cache_text:
        raise ValueError("shared pilot refuses a formal cache path")
    if not args.manifest.name.endswith("pilot_shared_groups.jsonl"):
        raise ValueError("shared pilot requires pilot_shared_groups.jsonl")

    groups = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not groups:
        raise ValueError("empty shared pilot manifest")
    for group in groups:
        members = group.get("members")
        if not isinstance(members, list) or len(members) != 2:
            raise ValueError("each pilot group must have exactly two members")
        roles = {member.get("role") for member in members}
        if roles != {"source", "target"}:
            raise ValueError(f"invalid roles in group {group.get('group_id')}: {roles}")

    args.cache.mkdir(parents=True, exist_ok=True)
    manifest_sha = _sha256(args.manifest)
    extractor_sha = _sha256(Path(__file__))
    progress_path = args.cache / "progress.json"
    failures_path = args.cache / "failures.json"
    failures: list[dict[str, Any]] = []

    def write_progress(completed: int) -> None:
        _atomic_json(
            progress_path,
            {
                "total_groups": len(groups),
                "completed_groups": completed,
                "failed_groups": len(failures),
                "cache": str(args.cache),
                "manifest_sha256": manifest_sha,
                "extractor_sha256": extractor_sha,
                "seed": args.seed,
                "steps": args.steps,
                "inner_steps": args.inner_steps,
                "guidance_scale": args.guidance_scale,
                "attention_resolution": args.resolution,
                "updated_at_unix": time.time(),
                "pilot_only": True,
            },
        )
        _atomic_json(failures_path, failures)

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
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
        "",
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.cuda()

    completed = 0
    write_progress(0)
    for group in groups:
        group_id = group["group_id"]
        members = group["members"]
        store = None
        try:
            reference = members[0]
            if any(
                member.get("image_path") != reference.get("image_path")
                or member.get("caption") != reference.get("caption")
                for member in members
            ):
                raise ValueError(f"shared group has mismatched image/caption: {group_id}")
            image_path = args.dataset_root / reference["image_path"]
            image = Image.open(image_path).convert("RGB").resize((512, 512))
            pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).float()
            pixels = pixels.div(127.5).sub(1).to("cuda", dtype=torch.float32)
            with torch.no_grad():
                latent = pipe.vae.encode(pixels).latent_dist.mean * pipe.vae.config.scaling_factor

            tokens = pipe.tokenizer(
                reference["caption"],
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                conditional = pipe.text_encoder(tokens.input_ids.cuda())[0]
                unconditional = pipe.text_encoder(empty)[0]
            trajectory = invert_ddim(
                pipe.unet,
                inverse,
                latent,
                conditional,
                steps=args.steps,
                device="cuda",
            )
            optimized = optimize_null_text(
                pipe.unet,
                forward,
                trajectory,
                unconditional,
                conditional,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                inner_steps=args.inner_steps,
                device="cuda",
            )

            # The hook is installed only for the final reconstruction loop.
            store = install_cross_attention_capture(pipe.unet, conditional_cfg_half=True)
            current = trajectory[-1]
            forward.set_timesteps(args.steps, device="cuda")
            with torch.no_grad():
                for timestep, uncond in zip(forward.timesteps, optimized.unconditional_embeddings):
                    pair = torch.cat([current, current])
                    output = pipe.unet(
                        pair,
                        timestep,
                        encoder_hidden_states=torch.cat([uncond, conditional]),
                    ).sample
                    uncond_noise, cond_noise = output.chunk(2)
                    current = forward.step(
                        uncond_noise + args.guidance_scale * (cond_noise - uncond_noise),
                        timestep,
                        current,
                    ).prev_sample

            tensors = store.maps.get(args.resolution, [])
            if not tensors:
                raise ValueError(f"no attention tensors at resolution {args.resolution}")
            if args.resolution == 16 and len(tensors) != args.steps * 5:
                raise ValueError("SD1.5 capture count differs from audited 100-tensor layout")
            attention = torch.cat(tensors, dim=0).mean(dim=0)
            capture_digest = hashlib.sha256(
                attention.detach().cpu().numpy().tobytes()
            ).hexdigest()
            if not np.isfinite(attention.detach().cpu().numpy()).all():
                raise ValueError("nonfinite shared attention tensor")

            for member in members:
                positions = _phrase_positions(
                    pipe.tokenizer,
                    reference["caption"],
                    member["phrase"],
                    member.get("phrase_occurrence", "first"),
                )
                spatial = attention[:, positions].mean(dim=-1).reshape(
                    args.resolution, args.resolution
                ).detach().cpu().numpy()
                if not np.isfinite(spatial).all() or (spatial < 0).any() or spatial.sum() <= 0:
                    raise ValueError(f"invalid map for {member['sample_id']}")
                spatial /= max(float(spatial.sum()), 1e-12)
                key = member["sample_id"].replace(":", "_")
                map_path = args.cache / f"{key}.npy"
                metadata_path = args.cache / f"{key}.json"
                temporary_map = map_path.with_suffix(".npy.tmp")
                with temporary_map.open("wb") as handle:
                    np.save(handle, spatial)
                temporary_map.replace(map_path)
                _atomic_json(
                    metadata_path,
                    {
                        "method": "ddim_nulltext_fp32_reconstruction_v3",
                        "capture_stage": "final_reconstruction_only",
                        "pilot_only": True,
                        "shared_group_id": group_id,
                        "shared_capture_digest": capture_digest,
                        "sample_id": member["sample_id"],
                        "caption": member["caption"],
                        "phrase": member["phrase"],
                        "phrase_token_positions": positions,
                        "phrase_occurrence": member.get("phrase_occurrence", "first"),
                        "role": member["role"],
                        "inversion_steps": args.steps,
                        "inner_steps": args.inner_steps,
                        "guidance_scale": args.guidance_scale,
                        "attention_resolution": args.resolution,
                        "attention_tensors": len(tensors),
                        "seed": args.seed,
                        "mean_reconstruction_mse": sum(optimized.reconstruction_losses)
                        / len(optimized.reconstruction_losses),
                        "map_path": str(map_path),
                        "map_sha256": _sha256(map_path),
                        "image_sha256": _sha256(image_path),
                        "instruction_sha256": hashlib.sha256(
                            member.get("instruction", member["caption"]).encode()
                        ).hexdigest(),
                        "manifest_sha256": manifest_sha,
                        "extractor_sha256": extractor_sha,
                        "teacher_frozen": True,
                        "numerics": NUMERICS,
                        "source_provenance": {
                            key: member[key]
                            for key in (
                                "episode_id",
                                "timestep",
                                "camera",
                                "split",
                                "role",
                                "instruction",
                            )
                            if key in member
                        },
                    },
                )
            completed += 1
        except Exception as exc:
            failures.append({"group_id": group_id, "error": repr(exc)})
            print(json.dumps(failures[-1]), flush=True)
        finally:
            if store is not None:
                store.maps.clear()
            pipe.unet.set_attn_processor(dict(original_processors))
            write_progress(completed)

    print(
        json.dumps(
            {
                "total_groups": len(groups),
                "completed_groups": completed,
                "failed_groups": len(failures),
                "pilot_only": True,
            }
        )
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
