"""Audit a true language-phrase attribution path in OpenVLA/OFT.

The native OFT action objective supervises action tokens only.  This probe
scores the source and target noun phrases already present in the instruction
prompt, differentiates their teacher-forced log-probability with respect to
visual tokens, and checks that the resulting semantic loss reaches trainable
parameters.  It performs no optimizer step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np


EXPECTED_SEMANTIC_SHA256 = (
    "d7660c9c8286483ebeeafee2d8ff2100dce99ddc9151b35f4beb03c3e3fb87fe"
)


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


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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
        "p1-report",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--temperature", type=float, default=1.25)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")

    from vla_attention.oft_preflight import require_gpu_window, require_p1, verify_snapshot

    require_gpu_window(args.gpu_window)
    source = verify_snapshot(args.model, args.source_lock)
    require_gpu_window(args.gpu_window)
    require_p1(
        args.p1_report,
        source["revision"],
        sha(args.train_manifest),
        sha(args.statistics),
    )
    semantic_sha = sha(args.semantic_manifest)
    if semantic_sha != EXPECTED_SEMANTIC_SHA256:
        raise ValueError(f"unexpected semantic manifest SHA256: {semantic_sha}")
    semantic_rows = read_jsonl(args.semantic_manifest)
    semantic_by_key = {
        (row["episode_id"], int(row["timestep"]), row["camera"], row["role"]): row
        for row in semantic_rows
    }
    if len(semantic_by_key) != len(semantic_rows):
        raise ValueError("duplicate semantic cache keys")
    train_rows = read_jsonl(args.train_manifest)
    eval_rows = read_jsonl(args.eval_manifest)
    if {row["episode_id"] for row in train_rows} & {
        row["episode_id"] for row in eval_rows
    }:
        raise ValueError("train/eval episode overlap")

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector
    from vla_attention.adapters.oft_forward import forward_l1
    from vla_attention.benchmarks.oft_rlds import EpisodeDataset
    from vla_attention.losses import semantic_map_kl

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

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
    params = [
        parameter
        for module in (policy, head, proprio)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    train = EpisodeDataset(args.train_manifest, args.statistics, args.model)
    order = list(range(len(train)))
    random.Random(args.seed).shuffle(order)
    batch = train.collate([train[order[0]]])
    episode = batch["episode_ids"][0]
    timestep = int(batch["timesteps"][0])

    rows_by_role: dict[str, dict] = {}
    for camera in ("image", "wrist_image"):
        for role in ("source", "target"):
            row = semantic_by_key.get((episode, timestep, camera, role))
            if row is None:
                raise KeyError(f"missing semantic map {episode}/{timestep}/{camera}/{role}")
            rows_by_role[f"{camera}:{role}"] = row

    phrases = {
        role: rows_by_role[f"image:{role}"]["phrase"] for role in ("source", "target")
    }
    occurrences = {
        role: rows_by_role[f"image:{role}"].get("phrase_occurrence", "first")
        for role in ("source", "target")
    }

    captured: list[torch.Tensor] = []

    def capture_visual_tokens(_module, _inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        if torch.is_tensor(value):
            captured[:] = [value]
        return output

    hook = policy.base_model.model.vision_backbone.register_forward_hook(
        capture_visual_tokens
    )
    try:
        output = forward_l1(
            policy, head, proprio, batch, device="cuda", visual_token_count=patch_count
        )
    finally:
        hook.remove()
    if not captured or not captured[0].requires_grad:
        raise RuntimeError("visual backbone output is not differentiable")

    sequence = [int(value) for value in output["input_ids"][0].tolist()]
    phrase_positions: dict[str, list[int]] = {}
    phrase_scores = []
    for role in ("source", "target"):
        phrase_ids = train.tokenizer(
            phrases[role], add_special_tokens=False
        )["input_ids"]
        matches = find_matches(sequence, phrase_ids)
        if not matches:
            raise RuntimeError(
                f"phrase not found in OpenVLA prompt: role={role!r} phrase={phrases[role]!r}"
            )
        position = matches[-1] if occurrences[role] == "last" else matches[0]
        if not position or min(position) <= 0:
            raise RuntimeError("phrase position must have a preceding causal-logit position")
        phrase_positions[role] = position
        positions = torch.tensor([index - 1 for index in position], device="cuda")
        target_ids = torch.tensor(
            [sequence[index] for index in position], device="cuda"
        )
        selected_logits = output["logits"][0].index_select(0, positions).float()
        phrase_scores.append(
            F.log_softmax(selected_logits, dim=-1)
            .gather(-1, target_ids.unsqueeze(-1))
            .mean()
        )

    language_score = torch.stack(phrase_scores).mean()
    (gradient,) = torch.autograd.grad(
        language_score,
        captured[0],
        retain_graph=True,
        create_graph=True,
        allow_unused=False,
    )
    features = captured[0][:, :patch_count]
    gradient = gradient[:, :patch_count]
    language_map = (
        gradient.float().mul(features.float()).sum(dim=-1).abs().reshape(1, patch_count)
    )

    maps = []
    for camera in ("image", "wrist_image"):
        role_maps = []
        for role in ("source", "target"):
            row = rows_by_role[f"{camera}:{role}"]
            values = np.load(
                args.semantic_root / f"{row['sample_id']}.npy", allow_pickle=False
            )
            if values.shape != (16, 16) or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("invalid semantic cache tensor")
            role_maps.append(torch.from_numpy(values).float().reshape(-1))
        maps.append(torch.stack(role_maps).mean(0))
    teacher_map = torch.cat(maps).cuda().reshape(1, patch_count)
    semantic_loss = semantic_map_kl(
        language_map, teacher_map, teacher_temperature=args.temperature
    )
    if not semantic_loss.requires_grad:
        raise RuntimeError("semantic loss is detached from the phrase attribution graph")
    semantic_grads = torch.autograd.grad(
        semantic_loss,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    nonzero = [
        int(
            gradient is not None
            and torch.isfinite(gradient).all()
            and gradient.abs().sum() > 0
        )
        for gradient in semantic_grads
    ]
    norms = [
        float(gradient.detach().float().norm()) if gradient is not None else 0.0
        for gradient in semantic_grads
    ]
    report = {
        "status": "passed" if any(nonzero) else "failed",
        "purpose": "true VLA language phrase attribution differentiability gate; no optimizer step",
        "model_revision": source["revision"],
        "train_manifest_sha256": sha(args.train_manifest),
        "eval_manifest_sha256": sha(args.eval_manifest),
        "statistics_sha256": sha(args.statistics),
        "semantic_manifest_sha256": semantic_sha,
        "seed": args.seed,
        "temperature": args.temperature,
        "student_signal": "phrase-score log-probability gradient times visual-token activation",
        "phrase_scores": {role: float(score.detach()) for role, score in zip(("source", "target"), phrase_scores)},
        "phrase_positions": phrase_positions,
        "phrases": phrases,
        "phrase_occurrences": occurrences,
        "create_graph": True,
        "student_map_shape": list(language_map.shape),
        "teacher_map_shape": list(teacher_map.shape),
        "action_loss": float(output["loss"].detach()),
        "semantic_loss": float(semantic_loss.detach()),
        "trainable_parameter_count": len(params),
        "semantic_gradient_nonzero_parameter_count": sum(nonzero),
        "semantic_gradient_finite": all(
            gradient is None or bool(torch.isfinite(gradient).all())
            for gradient in semantic_grads
        ),
        "semantic_gradient_norm_max": max(norms, default=0.0),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "episode_id": episode,
        "timestep": timestep,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if report["status"] != "passed":
        raise RuntimeError("semantic phrase loss has no non-zero trainable gradient")


if __name__ == "__main__":
    main()
