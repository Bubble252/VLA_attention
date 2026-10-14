#!/usr/bin/env python3
"""P6 tiny-overfit action interface smoke.

This diagnostic intentionally trains on a tiny task0 subset selected from expert
replay positive-control records. It keeps the same OFT restore path as P6.2 but
does not require the tiny train manifest SHA to equal the original P1 report;
instead it records both the original P1 provenance and the tiny manifest SHA.

It is not a semantic method and not a formal benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def set_determinism(seed: int) -> None:
    import numpy as np
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_weights(text: str, n: int, name: str) -> list[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != n or any(v <= 0 for v in vals):
        raise ValueError(f"--{name} must contain {n} positive values")
    return vals


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    for name in ("model", "source-lock", "train-manifest", "eval-manifest", "statistics", "gpu-window", "p1-report", "b1-checkpoint", "output"):
        ap.add_argument(f"--{name}", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--action-dim-weights", default="1,1,1,2,3,2,1")
    ap.add_argument("--horizon-weights", default="2,1.5,1.25,1,1,0.8,0.8,0.8")
    ap.add_argument("--checkpoint-step", type=int, default=150)
    ap.add_argument("--translation-magnitude-weight", type=float, default=0.0, help="Optional L1 loss on per-horizon translation vector magnitude. Default 0 keeps historical behavior.")
    ap.add_argument("--rotation-magnitude-weight", type=float, default=0.0, help="Optional L1 loss on per-horizon rotation vector magnitude. Default 0 keeps historical behavior.")
    ap.add_argument("--train-policy", action="store_true", help="If set, also train LoRA policy adapter. Default freezes policy and trains head/proprio only.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.steps <= 0 or args.lr <= 0 or args.weight_decay < 0 or args.translation_magnitude_weight < 0 or args.rotation_magnitude_weight < 0:
        raise ValueError("invalid optimization args")
    if not args.b1_checkpoint.is_dir():
        raise FileNotFoundError(args.b1_checkpoint)
    dim_weights_list = parse_weights(args.action_dim_weights, 7, "action-dim-weights")
    horizon_weights_list = parse_weights(args.horizon_weights, 8, "horizon-weights")

    from vla_attention.oft_preflight import require_gpu_window, verify_snapshot

    require_gpu_window(args.gpu_window)
    source = verify_snapshot(args.model, args.source_lock)
    train_sha = sha(args.train_manifest)
    eval_sha = sha(args.eval_manifest)
    stats_sha = sha(args.statistics)
    p1_report = read_jsonl(args.p1_report) if args.p1_report.suffix == ".jsonl" else None
    if p1_report is None:
        p1 = json.loads(args.p1_report.read_text())
        if p1.get("status") != "passed" or p1.get("model_revision") != source["revision"] or p1.get("statistics_sha256") != stats_sha:
            raise ValueError("P1 report does not match model/statistics provenance")
        if p1.get("action_shape") != [1, 8, 7] or not p1.get("gradient_finite") or not p1.get("gradient_nonzero"):
            raise ValueError("P1 report lacks required action/gradient gates")
        p1_original_train_manifest_sha256 = p1.get("train_manifest_sha256")
    else:
        p1_original_train_manifest_sha256 = None

    train_rows = read_jsonl(args.train_manifest)
    eval_rows = read_jsonl(args.eval_manifest)
    overlap_episode_ids = sorted({r["episode_id"] for r in train_rows} & {r["episode_id"] for r in eval_rows})
    # Tiny overfit is intentionally evaluated on the same few episodes.
    # Formal held-out evaluators must continue to reject overlap; this script is
    # only an action-interface memorization gate.

    import numpy as np
    import torch
    from peft import PeftModel
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector
    from vla_attention.adapters.oft_forward import forward_l1
    from vla_attention.benchmarks.oft_rlds import EpisodeDataset

    set_determinism(args.seed)
    config = OpenVLAConfig.from_pretrained(args.model, local_files_only=True)
    base = OpenVLAForActionPrediction.from_pretrained(
        args.model,
        config=config,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda()
    base.vision_backbone.set_num_images_in_input(2)
    patch_count = base.vision_backbone.get_num_patches() * 2
    policy = PeftModel.from_pretrained(base, args.b1_checkpoint / "adapter", is_trainable=bool(args.train_policy))
    head = L1RegressionActionHead(input_dim=base.llm_dim, hidden_dim=base.llm_dim).cuda().bfloat16()
    proprio = ProprioProjector(llm_dim=base.llm_dim, proprio_dim=8).cuda().bfloat16()
    state = torch.load(args.b1_checkpoint / "modules.pt", map_location="cpu", weights_only=False)
    head.load_state_dict(state["head"])
    proprio.load_state_dict(state["proprio"])

    if not args.train_policy:
        for p in policy.parameters():
            p.requires_grad_(False)

    train = EpisodeDataset(args.train_manifest, args.statistics, args.model)
    evaluation = EpisodeDataset(args.eval_manifest, args.statistics, args.model)
    order = list(range(len(train)))
    random.Random(args.seed).shuffle(order)
    eval_probe_indices = [
        next(index for index, (episode_index, _timestep) in enumerate(evaluation.index) if episode_index == row_index)
        for row_index in range(len(evaluation.rows))
    ]

    modules = [policy, head, proprio]
    trainable_modules = [head, proprio] if not args.train_policy else modules
    params = [p for m in trainable_modules for p in m.parameters() if p.requires_grad]
    if not params:
        raise ValueError("no trainable parameters")
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    dim_weights = torch.tensor(dim_weights_list, dtype=torch.float32, device="cuda").reshape(1, 1, 7)
    horizon_weights = torch.tensor(horizon_weights_list, dtype=torch.float32, device="cuda").reshape(1, 8, 1)

    args.output.mkdir(parents=True)
    metadata = {
        "status": "running",
        "experiment": "P6-tiny-overfit-action-interface-smoke",
        "base": "P6_B1_checkpoint_10080",
        "b1_checkpoint": str(args.b1_checkpoint),
        "b1_checkpoint_json_sha256": sha(args.b1_checkpoint / "checkpoint.json"),
        "b1_modules_sha256": sha(args.b1_checkpoint / "modules.pt"),
        "model_revision": source["revision"],
        "seed": args.seed,
        "steps": args.steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "action_dim_weights": dim_weights_list,
        "horizon_weights": horizon_weights_list,
        "train_policy": bool(args.train_policy),
        "trainable_scope": "policy_lora+head+proprio" if args.train_policy else "head+proprio_only_policy_frozen",
        "loss": "weighted_l1 + optional translation/rotation magnitude L1 over native 8x7 action chunk",
        "translation_magnitude_weight": args.translation_magnitude_weight,
        "rotation_magnitude_weight": args.rotation_magnitude_weight,
        "semantic_loss": None,
        "not_formal_benchmark": True,
        "train_manifest_sha256": train_sha,
        "eval_manifest_sha256": eval_sha,
        "statistics_sha256": stats_sha,
        "purpose": "tiny same-task action-head/proprio overfit gate before same-init rollout",
        "p1_original_train_manifest_sha256": p1_original_train_manifest_sha256,
        "tiny_manifest_intentional": True,
        "train_eval_overlap_episode_ids": overlap_episode_ids,
        "train_eval_overlap_allowed_only_for_tiny_overfit": True,
    }
    (args.output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "git_sha.txt").write_text(json.dumps({"git_sha": os.environ.get("VLA_GIT_SHA", "unrecorded")}, indent=2) + "\n")

    def save_checkpoint(step: int) -> Path:
        ckpt = args.output / f"checkpoint_{step:05d}"
        tmp = args.output / f".checkpoint_{step:05d}.tmp"
        if tmp.exists():
            import shutil
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        modes = [m.training for m in modules]
        for m in modules:
            m.eval()
        restore_probe = []
        with torch.no_grad():
            for index in eval_probe_indices:
                batch = evaluation.collate([evaluation[index]])
                out = forward_l1(policy, head, proprio, batch, device="cuda", visual_token_count=patch_count)
                restore_probe.append({
                    "episode_id": batch["episode_ids"][0],
                    "timestep": int(batch["timesteps"][0]),
                    "prediction": out["prediction"].float().cpu().tolist(),
                    "native_l1": float(out["loss"].detach()),
                })
        for m, mode in zip(modules, modes):
            m.train(mode)
        policy.save_pretrained(tmp / "adapter")
        torch.save({"head": head.state_dict(), "proprio": proprio.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": step, "metadata": metadata}, tmp / "modules.pt")
        (tmp / "restore_probe.json").write_text(json.dumps(restore_probe, indent=2) + "\n")
        (tmp / "checkpoint.json").write_text(json.dumps({"step": step, "config_sha256": sha(args.output / "config.json"), "train_manifest_sha256": train_sha, "eval_manifest_sha256": eval_sha, "model_revision": source["revision"], "base_b1_checkpoint": str(args.b1_checkpoint), "optimizer_included": True, "restore_probe_count": len(restore_probe), "trainable_scope": metadata["trainable_scope"]}, indent=2) + "\n")
        tmp.replace(ckpt)
        return ckpt

    metrics_path = args.output / "train_metrics.jsonl"
    failures_path = args.output / "failure_episodes.jsonl"
    started = time.time()
    for step in range(args.steps):
        batch = None
        try:
            batch = train.collate([train[order[step % len(order)]]])
            for m in modules:
                m.train()
            if not args.train_policy:
                policy.eval()
            torch.cuda.reset_peak_memory_stats()
            optimizer.zero_grad(set_to_none=True)
            out = forward_l1(policy, head, proprio, batch, device="cuda", visual_token_count=patch_count)
            pred = out["prediction"].float()
            target = batch["actions"].to(device="cuda", dtype=pred.dtype)
            abs_err = (pred - target).abs()
            weighted_l1_loss = (abs_err * dim_weights * horizon_weights).mean()
            pred_trans_mag = torch.linalg.vector_norm(pred[:, :, :3], ord=2, dim=-1)
            target_trans_mag = torch.linalg.vector_norm(target[:, :, :3], ord=2, dim=-1)
            pred_rot_mag = torch.linalg.vector_norm(pred[:, :, 3:6], ord=2, dim=-1)
            target_rot_mag = torch.linalg.vector_norm(target[:, :, 3:6], ord=2, dim=-1)
            translation_magnitude_loss = (pred_trans_mag - target_trans_mag).abs().mean()
            rotation_magnitude_loss = (pred_rot_mag - target_rot_mag).abs().mean()
            weighted_action_loss = (
                weighted_l1_loss
                + args.translation_magnitude_weight * translation_magnitude_loss
                + args.rotation_magnitude_weight * rotation_magnitude_loss
            )
            native_l1 = abs_err.mean()
            first_action_l1 = abs_err[:, 0, :].mean()
            future_action_l1 = abs_err[:, 1:, :].mean()
            per_dim_l1 = abs_err.reshape(-1, 7).mean(dim=0)
            per_horizon_l1 = abs_err.mean(dim=(0, 2))
            per_dim_pred_abs = pred.abs().reshape(-1, 7).mean(dim=0)
            per_dim_target_abs = target.abs().reshape(-1, 7).mean(dim=0)
            if not torch.isfinite(weighted_action_loss):
                raise FloatingPointError("non-finite weighted action loss")
            weighted_action_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            rec = {
                "optimizer_step": step + 1,
                "max_steps": args.steps,
                "progress_fraction": (step + 1) / args.steps,
                "episode_id": batch["episode_ids"][0],
                "timestep": int(batch["timesteps"][0]),
                "weighted_action_loss": float(weighted_action_loss.detach()),
                "weighted_l1_loss": float(weighted_l1_loss.detach()),
                "translation_magnitude_loss": float(translation_magnitude_loss.detach()),
                "rotation_magnitude_loss": float(rotation_magnitude_loss.detach()),
                "native_l1": float(native_l1.detach()),
                "first_action_l1": float(first_action_l1.detach()),
                "future_action_l1": float(future_action_l1.detach()),
                "per_dim_l1": [float(x) for x in per_dim_l1.detach().cpu()],
                "per_horizon_l1": [float(x) for x in per_horizon_l1.detach().cpu()],
                "per_dim_pred_abs": [float(x) for x in per_dim_pred_abs.detach().cpu()],
                "per_dim_target_abs": [float(x) for x in per_dim_target_abs.detach().cpu()],
                "translation_magnitude_weight": args.translation_magnitude_weight,
                "rotation_magnitude_weight": args.rotation_magnitude_weight,
                "gradient_norm_pre_clip": float(grad_norm),
                "learning_rate": scheduler.get_last_lr()[0],
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "finite": True,
                "not_formal_benchmark": True,
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if step + 1 == args.checkpoint_step or step + 1 == args.steps:
                save_checkpoint(step + 1)
            print(json.dumps(rec, ensure_ascii=False), flush=True)
        except Exception as exc:
            failure = {"optimizer_step": step + 1, "episode_id": batch["episode_ids"][0] if batch else "unknown", "reason": repr(exc)}
            with failures_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(failure, ensure_ascii=False) + "\n")
            raise

    metadata["status"] = "passed"
    metadata["completed_at_unix"] = time.time()
    (args.output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    final = {"status": "passed", "steps": args.steps, "base": "B1", "checkpoint": str(args.output / f"checkpoint_{args.steps:05d}"), "elapsed_seconds": time.time() - started, "not_formal_benchmark": True, "tiny_overfit": True}
    (args.output / "final.json").write_text(json.dumps(final, indent=2) + "\n")
    print("P6_TINY_OVERFIT_SMOKE_PASSED", flush=True)


if __name__ == "__main__":
    main()
