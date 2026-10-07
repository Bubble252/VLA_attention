"""Unified OpenVLA/OFT offline B0--B4 runner.

This is the first runner that keeps the frozen B0--B4 matrix, native action
loss, phrase-score A_lang, differentiable A_act, cache provenance, checkpoint
cadence and per-step artifacts in one process.  It is offline action training
only; it does not launch LIBERO rollouts.

The formal candidate uses one transition per optimizer step.  This is
intentional: B2--B4 use create_graph=True attribution and must not average
different image/caption evidence maps inside an un-audited batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


EXPECTED_SEMANTIC_SHA256 = (
    "d7660c9c8286483ebeeafee2d8ff2100dce99ddc9151b35f4beb03c3e3fb87fe"
)
EXPECTED_RADIO_SHA256 = (
    "2c47200466c63e43c187b6abad68b533ec07ffd3f9ad722c77537672179478ac"
)
GROUPS = {
    "B0": {"retention": False, "semantic": False, "containment": False},
    "B1": {"retention": True, "semantic": False, "containment": False},
    "B2": {"retention": False, "semantic": True, "containment": False},
    "B3": {"retention": True, "semantic": True, "containment": False},
    "B4": {"retention": True, "semantic": True, "containment": True},
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def find_matches(sequence: list[int], pattern: list[int]) -> list[list[int]]:
    if not pattern:
        return []
    return [
        list(range(index, index + len(pattern)))
        for index in range(len(sequence) - len(pattern) + 1)
        if sequence[index : index + len(pattern)] == pattern
    ]


def phrase_score(
    *,
    logits: Any,
    input_ids: Any,
    tokenizer: Any,
    phrase: str,
    occurrence: str,
):
    import torch
    import torch.nn.functional as F

    sequence = [int(value) for value in input_ids[0].tolist()]
    phrase_ids = tokenizer(phrase, add_special_tokens=False)["input_ids"]
    matches = find_matches(sequence, phrase_ids)
    if not matches:
        raise RuntimeError(f"phrase not found in prompt: {phrase!r}")
    positions = matches[-1] if occurrence == "last" else matches[0]
    if min(positions) <= 0:
        raise RuntimeError(f"phrase has no causal predecessor: {phrase!r}")
    predecessor = torch.tensor([index - 1 for index in positions], device=logits.device)
    targets = torch.tensor([sequence[index] for index in positions], device=logits.device)
    selected = logits[0].index_select(0, predecessor).float()
    score = F.log_softmax(selected, dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).mean()
    return score, positions


def set_determinism(seed: int, *, require_second_order: bool) -> None:
    import numpy as np
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if require_second_order:
        # The pinned OpenVLA/OFT graph uses SDPA.  Flash and mem-efficient
        # kernels have no second derivative on the installed PyTorch build.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "model",
        "source-lock",
        "train-manifest",
        "eval-manifest",
        "statistics",
        "gpu-window",
        "semantic-root",
        "semantic-manifest",
        "radio-root",
        "radio-manifest",
        "p1-report",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--group", choices=tuple(GROUPS), required=True)
    parser.add_argument("--seed", type=int, choices=(17, 29, 41), default=17)
    parser.add_argument("--steps", type=int, default=10080)
    parser.add_argument("--lambda-semantic", type=float, default=0.10)
    parser.add_argument("--lambda-retention", type=float, default=0.05)
    parser.add_argument("--lambda-containment", type=float, default=0.02)
    parser.add_argument("--retention-warmup-steps", type=int, default=500)
    parser.add_argument("--containment-warmup-steps", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=1.25)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--checkpoint-fractions", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--code-bundle-sha256", default="unrecorded")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if min(args.lambda_semantic, args.lambda_retention, args.lambda_containment) < 0:
        raise ValueError("loss weights must be non-negative")
    if args.temperature <= 0 or args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("temperature/lr/weight_decay are invalid")
    if min(args.retention_warmup_steps, args.containment_warmup_steps) < 0:
        raise ValueError("warmup steps must be non-negative")
    if args.output.exists() and args.resume_from is None:
        raise FileExistsError(args.output)
    if args.resume_from is not None and not args.resume_from.is_dir():
        raise FileNotFoundError(args.resume_from)
    switches = GROUPS[args.group]
    second_order = switches["semantic"] or switches["containment"]
    fractions = tuple(
        float(value)
        for value in args.checkpoint_fractions.split(",")
        if value.strip()
    )
    if not fractions or any(not 0.0 < value <= 1.0 for value in fractions):
        raise ValueError("checkpoint fractions must lie in (0,1]")
    checkpoint_steps = sorted({max(1, math.ceil(args.steps * value)) for value in fractions})

    from vla_attention.oft_preflight import require_gpu_window, require_p1, verify_snapshot

    require_gpu_window(args.gpu_window)
    source = verify_snapshot(args.model, args.source_lock)
    require_gpu_window(args.gpu_window)
    train_sha = sha(args.train_manifest)
    eval_sha = sha(args.eval_manifest)
    stats_sha = sha(args.statistics)
    require_p1(args.p1_report, source["revision"], train_sha, stats_sha)

    semantic_rows = read_jsonl(args.semantic_manifest)
    semantic_by_key = {
        (row["episode_id"], int(row["timestep"]), row["camera"], row["role"]): row
        for row in semantic_rows
    }
    if len(semantic_by_key) != len(semantic_rows):
        raise ValueError("duplicate semantic cache keys")
    radio_rows = read_jsonl(args.radio_manifest)
    radio_by_key = {
        (row["episode_id"], int(row["timestep"]), row["camera"]): row
        for row in radio_rows
    }
    if len(radio_by_key) != len(radio_rows):
        raise ValueError("duplicate radio cache keys")
    semantic_sha = sha(args.semantic_manifest)
    radio_sha = sha(args.radio_manifest)
    if switches["semantic"] and semantic_sha != EXPECTED_SEMANTIC_SHA256:
        raise ValueError(f"unexpected semantic manifest SHA256: {semantic_sha}")
    if switches["retention"] and radio_sha != EXPECTED_RADIO_SHA256:
        raise ValueError(f"unexpected radio manifest SHA256: {radio_sha}")

    train_rows = read_jsonl(args.train_manifest)
    eval_rows = read_jsonl(args.eval_manifest)
    train_ids = {row["episode_id"] for row in train_rows}
    eval_ids = {row["episode_id"] for row in eval_rows}
    if train_ids & eval_ids:
        raise ValueError("train/eval episode overlap")

    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector
    from vla_attention.adapters.oft_forward import forward_l1
    from vla_attention.benchmarks.oft_rlds import EpisodeDataset
    from vla_attention.losses import (
        action_evidence_containment,
        cosine_feature_retention,
        semantic_map_kl,
    )

    # Use one attention backend for every B0--B4 group.  This avoids making
    # the matrix compare flash attention in B0/B1 with math SDPA in B2--B4.
    # Math SDPA is required by the second-order attribution groups and is
    # still within the single-GPU memory budget for B0/B1.
    set_determinism(args.seed, require_second_order=True)
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
    if args.resume_from is not None:
        from peft import PeftModel

        policy = PeftModel.from_pretrained(
            base, args.resume_from / "adapter", is_trainable=True
        )
    else:
        policy = get_peft_model(
            base,
            LoraConfig(
                r=32,
                lora_alpha=16,
                lora_dropout=0.0,
                target_modules="all-linear",
                init_lora_weights="gaussian",
            ),
        )
    head = L1RegressionActionHead(
        input_dim=base.llm_dim, hidden_dim=base.llm_dim
    ).cuda().bfloat16()
    proprio = ProprioProjector(llm_dim=base.llm_dim, proprio_dim=8).cuda().bfloat16()
    train = EpisodeDataset(args.train_manifest, args.statistics, args.model)
    evaluation = EpisodeDataset(args.eval_manifest, args.statistics, args.model)
    if len(train) < 1:
        raise ValueError("empty train transition dataset")
    if not evaluation.rows:
        raise ValueError("empty offline evaluation manifest")
    order = list(range(len(train)))
    random.Random(args.seed).shuffle(order)
    eval_probe_indices = [
        next(
            index
            for index, (episode_index, _timestep) in enumerate(evaluation.index)
            if episode_index == row_index
        )
        for row_index in range(len(evaluation.rows))
    ]

    projector = None
    if switches["retention"]:
        projector = torch.nn.Linear(
            base.llm_dim,
            1024,
            bias=False,
            dtype=torch.bfloat16,
            device="cuda",
        )
    modules = [policy, head, proprio]
    if projector is not None:
        modules.append(projector)
    params = [
        parameter
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps
    )

    if args.resume_from is None:
        args.output.mkdir(parents=True)
    metadata = {
        "status": "running",
        "experiment": f"openvla-oft-offline-{args.group.lower()}",
        "group": args.group,
        "seed": args.seed,
        "steps": args.steps,
        "effective_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "dataset_size": len(train),
        "eval_probe_count": len(eval_probe_indices),
        "dataset_passes": math.ceil(args.steps / len(train)),
        "model_revision": source["revision"],
        "train_manifest_sha256": train_sha,
        "eval_manifest_sha256": eval_sha,
        "statistics_sha256": stats_sha,
        "semantic_manifest_sha256": semantic_sha,
        "radio_manifest_sha256": radio_sha,
        "matrix": switches,
        "action_loss": "native_mean_L1_on_fixed_native_action_target",
        "student_language_signal": (
            "phrase_score_gradient_times_activation"
            if switches["semantic"]
            else None
        ),
        "student_action_signal": (
            "action_loss_gradient_times_activation"
            if switches["containment"]
            else None
        ),
        "attribution_create_graph": second_order,
        "attention_backend": "math_sdpa",
        "semantic_temperature": args.temperature if switches["semantic"] else None,
        "lambda_semantic": args.lambda_semantic if switches["semantic"] else 0.0,
        "lambda_retention": args.lambda_retention if switches["retention"] else 0.0,
        "lambda_containment": args.lambda_containment if switches["containment"] else 0.0,
        "retention_warmup_steps": args.retention_warmup_steps if switches["retention"] else 0,
        "containment_warmup_steps": args.containment_warmup_steps if switches["containment"] else 0,
        "code_bundle_sha256": args.code_bundle_sha256,
        "checkpoint_steps": checkpoint_steps,
        "purpose": "formal offline candidate; no rollout success claim",
    }
    config_path = args.output / "config.json"
    if args.resume_from is None:
        config_path.write_text(json.dumps(metadata, indent=2) + "\n")
        (args.output / "git_sha.txt").write_text(
            json.dumps(
                {
                    "git_sha": os.environ.get("VLA_GIT_SHA", "unrecorded"),
                    "code_bundle_sha256": args.code_bundle_sha256,
                },
                indent=2,
            )
            + "\n"
        )
        (args.output / "manifest_sha256.json").write_text(
            json.dumps(
                {
                    "train": train_sha,
                    "eval": eval_sha,
                    "statistics": stats_sha,
                    "semantic": semantic_sha,
                    "radio": radio_sha,
                },
                indent=2,
            )
            + "\n"
        )
    else:
        prior_config = json.loads(config_path.read_text())
        stable_metadata = {key: value for key, value in metadata.items() if key != "status"}
        prior_stable = {key: value for key, value in prior_config.items() if key not in {"status", "completed_at_unix"}}
        if prior_stable != stable_metadata:
            raise ValueError("resume config/provenance mismatch")

    def load_teacher_maps(batch: dict[str, Any]):
        episode = batch["episode_ids"][0]
        timestep = int(batch["timesteps"][0])
        by_role: dict[str, list[torch.Tensor]] = {"source": [], "target": []}
        for camera in ("image", "wrist_image"):
            for role in ("source", "target"):
                row = semantic_by_key.get((episode, timestep, camera, role))
                if row is None:
                    raise KeyError(f"missing semantic cache: {episode}/{timestep}/{camera}/{role}")
                values = np.load(
                    args.semantic_root / f"{row['sample_id']}.npy",
                    allow_pickle=False,
                )
                if (
                    values.shape != (16, 16)
                    or not np.isfinite(values).all()
                    or (values < 0).any()
                ):
                    raise ValueError("invalid semantic cache tensor")
                by_role[role].append(torch.from_numpy(values).float().reshape(-1))
        source_map = torch.cat(by_role["source"]).cuda().reshape(1, patch_count)
        target_map = torch.cat(by_role["target"]).cuda().reshape(1, patch_count)
        return source_map, target_map

    def load_radio(batch: dict[str, Any]):
        episode = batch["episode_ids"][0]
        timestep = int(batch["timesteps"][0])
        values = []
        for camera in ("image", "wrist_image"):
            row = radio_by_key.get((episode, timestep, camera))
            if row is None:
                raise KeyError(f"missing radio cache: {episode}/{timestep}/{camera}")
            tensor = np.load(args.radio_root / row["output_file"], allow_pickle=False)
            if tensor.shape != (256, 1024) or not np.isfinite(tensor).all():
                raise ValueError("invalid C-RADIO cache tensor")
            values.append(torch.from_numpy(tensor))
        return torch.cat(values, dim=0).unsqueeze(0).cuda().float()

    def save_checkpoint(step: int) -> Path:
        checkpoint = args.output / f"checkpoint_{step:05d}"
        temporary = args.output / f".checkpoint_{step:05d}.tmp"
        if temporary.exists():
            import shutil

            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        training_modes = [module.training for module in modules]
        for module in modules:
            module.eval()
        restore_probe = []
        with torch.no_grad():
            for index in eval_probe_indices:
                probe_batch = evaluation.collate([evaluation[index]])
                probe_output = forward_l1(
                    policy,
                    head,
                    proprio,
                    probe_batch,
                    device="cuda",
                    visual_token_count=patch_count,
                )
                restore_probe.append(
                    {
                        "episode_id": probe_batch["episode_ids"][0],
                        "timestep": int(probe_batch["timesteps"][0]),
                        "prediction": probe_output["prediction"].float().cpu().tolist(),
                        "action_l1": float(probe_output["loss"].detach()),
                    }
                )
        for module, was_training in zip(modules, training_modes):
            module.train(was_training)
        policy.save_pretrained(temporary / "adapter")
        (temporary / "restore_probe.json").write_text(
            json.dumps(restore_probe, indent=2) + "\n"
        )
        torch.save(
            {
                "head": head.state_dict(),
                "proprio": proprio.state_dict(),
                "projector": None if projector is None else projector.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "metadata": metadata,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
            },
            temporary / "modules.pt",
        )
        (temporary / "checkpoint.json").write_text(
            json.dumps(
                {
                    "step": step,
                    "config_sha256": sha(config_path),
                    "train_manifest_sha256": train_sha,
                    "model_revision": source["revision"],
                    "semantic_gradient_mode": "create_graph_true" if second_order else "not_used",
                    "optimizer_included": True,
                    "scheduler_included": True,
                    "rng_included": True,
                    "restore_probe_count": len(restore_probe),
                },
                indent=2,
            )
            + "\n"
        )
        if checkpoint.exists():
            import shutil

            backup = args.output / f".checkpoint_{step:05d}.old"
            if backup.exists():
                shutil.rmtree(backup)
            checkpoint.replace(backup)
            try:
                temporary.replace(checkpoint)
            except Exception:
                backup.replace(checkpoint)
                raise
            shutil.rmtree(backup)
        else:
            temporary.replace(checkpoint)
        return checkpoint

    start_step = 0
    if args.resume_from is not None:
        checkpoint_json = json.loads(
            (args.resume_from / "checkpoint.json").read_text()
        )
        if checkpoint_json.get("train_manifest_sha256") != train_sha:
            raise ValueError("resume checkpoint train manifest mismatch")
        if checkpoint_json.get("model_revision") != source["revision"]:
            raise ValueError("resume checkpoint model revision mismatch")
        state = torch.load(
            args.resume_from / "modules.pt", map_location="cpu", weights_only=False
        )
        if state.get("metadata") != metadata:
            raise ValueError("resume checkpoint metadata mismatch")
        head.load_state_dict(state["head"])
        proprio.load_state_dict(state["proprio"])
        if projector is not None:
            if state.get("projector") is None:
                raise ValueError("resume checkpoint missing retention projector")
            projector.load_state_dict(state["projector"])
        elif state.get("projector") is not None:
            raise ValueError("resume checkpoint contains unexpected retention projector")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = int(state["step"])
        if not 0 < start_step < args.steps:
            raise ValueError("resume checkpoint step must lie in (0, max_steps)")
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        for module in modules:
            module.eval()
        restored_probe = []
        with torch.no_grad():
            for index in eval_probe_indices:
                probe_batch = evaluation.collate([evaluation[index]])
                probe_output = forward_l1(
                    policy,
                    head,
                    proprio,
                    probe_batch,
                    device="cuda",
                    visual_token_count=patch_count,
                )
                restored_probe.append(
                    {
                        "episode_id": probe_batch["episode_ids"][0],
                        "timestep": int(probe_batch["timesteps"][0]),
                        "prediction": probe_output["prediction"].float().cpu().tolist(),
                        "action_l1": float(probe_output["loss"].detach()),
                    }
                )
        saved_probe = json.loads(
            (args.resume_from / "restore_probe.json").read_text()
        )
        if len(saved_probe) != len(restored_probe):
            raise ValueError("resume restore probe coverage mismatch")
        max_prediction_error = 0.0
        for saved, restored in zip(saved_probe, restored_probe):
            if saved["episode_id"] != restored["episode_id"]:
                raise ValueError("resume restore episode mismatch")
            saved_prediction = np.asarray(saved["prediction"], dtype=np.float32)
            restored_prediction = np.asarray(
                restored["prediction"], dtype=np.float32
            )
            if saved_prediction.shape != restored_prediction.shape:
                raise ValueError("resume restore prediction shape mismatch")
            max_prediction_error = max(
                max_prediction_error,
                float(np.max(np.abs(saved_prediction - restored_prediction))),
            )
        (args.output / "restore_report.json").write_text(
            json.dumps(
                {
                    "status": "passed" if max_prediction_error == 0.0 else "failed",
                    "checkpoint": str(args.resume_from),
                    "saved_step": start_step,
                    "probe_count": len(restored_probe),
                    "max_prediction_error": max_prediction_error,
                },
                indent=2,
            )
            + "\n"
        )
        if max_prediction_error != 0.0:
            raise ValueError(
                f"resume restore prediction drift: {max_prediction_error}"
            )
        for module in modules:
            module.train()

    metrics_path = args.output / "train_metrics.jsonl"
    progress_path = args.output / "progress_summary.jsonl"
    failures_path = args.output / "failure_episodes.jsonl"
    started_at = time.time()
    for step in range(start_step, args.steps):
        batch = None
        try:
            batch = train.collate([train[order[step % len(order)]]])
            for module in modules:
                module.train()
            torch.cuda.reset_peak_memory_stats()
            optimizer.zero_grad(set_to_none=True)
            captured: list[torch.Tensor] = []

            def capture_visual(_module, _inputs, output):
                value = output[0] if isinstance(output, tuple) else output
                if torch.is_tensor(value):
                    captured[:] = [value]
                return output

            hook = None
            if switches["semantic"] or switches["containment"]:
                hook = policy.base_model.model.vision_backbone.register_forward_hook(
                    capture_visual
                )
            try:
                output = forward_l1(
                    policy,
                    head,
                    proprio,
                    batch,
                    device="cuda",
                    visual_token_count=patch_count,
                )
            finally:
                if hook is not None:
                    hook.remove()
            if not torch.isfinite(output["loss"]):
                raise FloatingPointError("non-finite action loss")

            retention_loss = output["loss"].new_zeros(())
            semantic_loss = output["loss"].new_zeros(())
            containment_loss = output["loss"].new_zeros(())
            retention_scale = min(
                1.0,
                (step + 1) / args.retention_warmup_steps
                if args.retention_warmup_steps
                else 1.0,
            )
            containment_scale = min(
                1.0,
                (step + 1) / args.containment_warmup_steps
                if args.containment_warmup_steps
                else 1.0,
            )
            action_map = None
            phrase_positions = {}
            if switches["retention"]:
                teacher_features = load_radio(batch)
                student_features = output["visual_and_proprio_features"][:, :patch_count]
                if student_features.shape[1] != teacher_features.shape[1]:
                    raise ValueError("retention patch count mismatch")
                retention_loss = cosine_feature_retention(
                    projector(student_features), teacher_features
                )

            source_teacher = target_teacher = None
            if switches["semantic"] or switches["containment"]:
                if not captured or not captured[0].requires_grad:
                    raise RuntimeError("visual tokens are not differentiable")
                visual_tokens = captured[0][:, :patch_count]
                if switches["containment"]:
                    (action_gradient,) = torch.autograd.grad(
                        output["loss"],
                        captured[0],
                        retain_graph=True,
                        create_graph=True,
                        allow_unused=False,
                    )
                    action_map = (
                        action_gradient[:, :patch_count]
                        .float()
                        .mul(visual_tokens.float())
                        .sum(dim=-1)
                        .abs()
                    )
                if switches["semantic"]:
                    source_teacher, target_teacher = load_teacher_maps(batch)
                    source_row = semantic_by_key[
                        (batch["episode_ids"][0], int(batch["timesteps"][0]), "image", "source")
                    ]
                    target_row = semantic_by_key[
                        (batch["episode_ids"][0], int(batch["timesteps"][0]), "image", "target")
                    ]
                    source_score, source_positions = phrase_score(
                        logits=output["logits"],
                        input_ids=output["input_ids"],
                        tokenizer=train.tokenizer,
                        phrase=source_row["phrase"],
                        occurrence=source_row.get("phrase_occurrence", "first"),
                    )
                    target_score, target_positions = phrase_score(
                        logits=output["logits"],
                        input_ids=output["input_ids"],
                        tokenizer=train.tokenizer,
                        phrase=target_row["phrase"],
                        occurrence=target_row.get("phrase_occurrence", "last"),
                    )
                    phrase_positions = {
                        "source": source_positions,
                        "target": target_positions,
                    }
                    (source_gradient,) = torch.autograd.grad(
                        source_score,
                        captured[0],
                        retain_graph=True,
                        create_graph=True,
                        allow_unused=False,
                    )
                    (target_gradient,) = torch.autograd.grad(
                        target_score,
                        captured[0],
                        retain_graph=True,
                        create_graph=True,
                        allow_unused=False,
                    )
                    source_map = (
                        source_gradient[:, :patch_count]
                        .float()
                        .mul(visual_tokens.float())
                        .sum(dim=-1)
                        .abs()
                    )
                    target_map = (
                        target_gradient[:, :patch_count]
                        .float()
                        .mul(visual_tokens.float())
                        .sum(dim=-1)
                        .abs()
                    )
                    semantic_loss = semantic_map_kl(
                        torch.cat([source_map, target_map], dim=0),
                        torch.cat([source_teacher, target_teacher], dim=0),
                        teacher_temperature=args.temperature,
                    )
                if switches["containment"]:
                    support = torch.maximum(source_teacher, target_teacher)
                    support = support / support.max(dim=-1, keepdim=True).values.clamp_min(1e-8)
                    normalized_action = action_map / action_map.sum(
                        dim=-1, keepdim=True
                    ).clamp_min(1e-8)
                    containment_loss = action_evidence_containment(
                        normalized_action, support
                    )

            total_loss = (
                output["loss"]
                + args.lambda_retention * retention_scale * retention_loss
                + args.lambda_semantic * semantic_loss
                + args.lambda_containment * containment_scale * containment_loss
            )
            if not torch.isfinite(total_loss):
                raise FloatingPointError("non-finite total loss")
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                params, args.max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            now = time.time()
            record = {
                "optimizer_step": step + 1,
                "max_steps": args.steps,
                "progress_fraction": (step + 1) / args.steps,
                "group": args.group,
                "seed": args.seed,
                "episode_id": batch["episode_ids"][0],
                "timestep": int(batch["timesteps"][0]),
                "action_loss": float(output["loss"].detach()),
                "retention_loss": float(retention_loss.detach()),
                "semantic_loss": float(semantic_loss.detach()),
                "containment_loss": float(containment_loss.detach()),
                "total_loss": float(total_loss.detach()),
                "effective_lambda_semantic": args.lambda_semantic,
                "effective_lambda_retention": args.lambda_retention * retention_scale,
                "effective_lambda_containment": args.lambda_containment * containment_scale,
                "retention_warmup_scale": retention_scale,
                "containment_warmup_scale": containment_scale,
                "gradient_norm_pre_clip": float(grad_norm),
                "learning_rate": scheduler.get_last_lr()[0],
                "step_seconds": now - started_at if step == 0 else now - previous_time,
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "finite": True,
                "attribution_create_graph": second_order,
                "phrase_positions": phrase_positions,
            }
            previous_time = now
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            if step + 1 in checkpoint_steps:
                save_checkpoint(step + 1)
            if step + 1 == 1 or step + 1 in {
                max(1, math.ceil(args.steps * fraction / 10.0) * 10)
                for fraction in [i / 10.0 for i in range(1, 11)]
            }:
                progress = {
                    "optimizer_step": step + 1,
                    "max_steps": args.steps,
                    "progress_fraction": (step + 1) / args.steps,
                    "group": args.group,
                    "seed": args.seed,
                    "successful_steps": step + 1,
                    "failed_steps": 0,
                    "train_manifest_sha256": train_sha,
                    "config_sha256": sha(args.output / "config.json"),
                }
                with progress_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(progress) + "\n")
            print(json.dumps(record, ensure_ascii=False), flush=True)
        except Exception as exc:
            episode = batch["episode_ids"][0] if batch is not None else "unknown"
            failure = {
                "optimizer_step": step + 1,
                "group": args.group,
                "seed": args.seed,
                "episode_id": episode,
                "reason": repr(exc),
            }
            with failures_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(failure, ensure_ascii=False) + "\n")
            raise

    metadata["status"] = "passed"
    metadata["completed_at_unix"] = time.time()
    (args.output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "final.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "group": args.group,
                "seed": args.seed,
                "steps": args.steps,
                "train_manifest_sha256": train_sha,
                "eval_manifest_sha256": eval_sha,
                "attribution_create_graph": second_order,
                "purpose": "offline action training; rollout not run",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"FORMAL_OFFLINE_{args.group}_SEED_{args.seed}_PASSED", flush=True)


if __name__ == "__main__":
    main()
