"""Correct P6 B3 smoke: retention + semantic attribution, without containment."""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path


EXPECTED_SEMANTIC_SHA256 = "d7660c9c8286483ebeeafee2d8ff2100dce99ddc9151b35f4beb03c3e3fb87fe"
EXPECTED_RADIO_SHA256 = "2c47200466c63e43c187b6abad68b533ec07ffd3f9ad722c77537672179478ac"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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
        "radio-root",
        "radio-manifest",
        "p1-report",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lambda-retention", type=float, default=0.05)
    parser.add_argument("--lambda-semantic", type=float, default=0.10)
    parser.add_argument("--temperature", type=float, choices=(1.0, 1.25), required=True)
    args = parser.parse_args()

    if not 20 <= args.steps <= 30:
        raise ValueError("B3 interface smoke must use 20-30 steps")
    if args.lambda_retention < 0 or args.lambda_semantic < 0:
        raise ValueError("loss weights must be non-negative")
    if args.output.exists():
        raise FileExistsError(args.output)
    if "B3" not in args.output.name and "b3" not in args.output.name:
        raise ValueError("output directory must identify B3")

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
        raise ValueError("duplicate C-RADIO cache keys")

    semantic_sha = sha(args.semantic_manifest)
    radio_sha = sha(args.radio_manifest)
    if semantic_sha != EXPECTED_SEMANTIC_SHA256:
        raise ValueError(f"unexpected semantic cache manifest SHA256: {semantic_sha}")
    if radio_sha != EXPECTED_RADIO_SHA256:
        raise ValueError(f"unexpected C-RADIO cache manifest SHA256: {radio_sha}")

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
    from vla_attention.losses import cosine_feature_retention, semantic_map_kl

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
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
    head = L1RegressionActionHead(input_dim=base.llm_dim, hidden_dim=base.llm_dim).cuda().bfloat16()
    proprio = ProprioProjector(llm_dim=base.llm_dim, proprio_dim=8).cuda().bfloat16()
    train = EpisodeDataset(args.train_manifest, args.statistics, args.model)
    order = list(range(len(train)))
    random.Random(args.seed).shuffle(order)
    projector = None
    optimizer = None
    records = []

    def load_semantic_support(batch):
        episode = batch["episode_ids"][0]
        timestep = int(batch["timesteps"][0])
        views = []
        for camera in ("image", "wrist_image"):
            roles = []
            for role in ("source", "target"):
                row = semantic_by_key.get((episode, timestep, camera, role))
                if row is None:
                    raise KeyError(f"missing semantic map: {episode}/{timestep}/{camera}/{role}")
                values = np.load(args.semantic_root / f"{row['sample_id']}.npy", allow_pickle=False)
                if values.shape != (16, 16) or not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError("invalid semantic cache tensor")
                roles.append(torch.from_numpy(values).float().reshape(-1))
            views.append(torch.stack(roles).mean(0))
        return torch.cat(views).cuda()

    def load_retention_target(batch):
        episode = batch["episode_ids"][0]
        timestep = int(batch["timesteps"][0])
        views = []
        for camera in ("image", "wrist_image"):
            row = radio_by_key.get((episode, timestep, camera))
            if row is None:
                raise KeyError(f"missing C-RADIO target: {episode}/{timestep}/{camera}")
            values = np.load(args.radio_root / row["output_file"], allow_pickle=False)
            if values.shape != (256, 1024) or not np.isfinite(values).all():
                raise ValueError("invalid C-RADIO cache tensor")
            views.append(torch.from_numpy(values))
        return torch.cat(views, dim=0).unsqueeze(0).cuda().float()

    for step in range(args.steps):
        batch = train.collate([train[order[step % len(order)]]])
        captured = []

        def capture_visual_tokens(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(value):
                value.retain_grad()
                captured[:] = [value]
            return output

        hook = policy.base_model.model.vision_backbone.register_forward_hook(capture_visual_tokens)
        try:
            output = forward_l1(
                policy, head, proprio, batch, device="cuda", visual_token_count=patch_count
            )
            output["loss"].backward(retain_graph=True)
        finally:
            hook.remove()
        if not captured or captured[0].grad is None:
            raise RuntimeError("action-loss gradient unavailable at visual tokens")

        features = captured[0][:, :patch_count]
        gradients = captured[0].grad[:, :patch_count]
        action_map = (
            gradients.float().mul(features.float()).sum(dim=-1).abs().reshape(1, patch_count)
        )
        teacher_map = load_semantic_support(batch).reshape(1, patch_count)
        teacher_features = load_retention_target(batch)
        if features.shape[1] != teacher_features.shape[1]:
            raise ValueError(
                f"retention patch count mismatch: {features.shape[1]} vs {teacher_features.shape[1]}"
            )

        if projector is None:
            projector = torch.nn.Linear(
                features.shape[-1],
                teacher_features.shape[-1],
                bias=False,
                dtype=torch.bfloat16,
                device="cuda",
            )
            params = [
                parameter
                for module in (policy, head, proprio, projector)
                for parameter in module.parameters()
                if parameter.requires_grad
            ]
            optimizer = torch.optim.AdamW(params, lr=5e-4, weight_decay=0.01)

        retention_loss = cosine_feature_retention(projector(features), teacher_features)
        semantic_loss = semantic_map_kl(
            action_map, teacher_map, teacher_temperature=args.temperature
        )
        total_loss = (
            output["loss"]
            + args.lambda_retention * retention_loss
            + args.lambda_semantic * semantic_loss
        )
        if not torch.isfinite(total_loss):
            raise FloatingPointError("non-finite B3 total loss")

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        record = {
            "step": step + 1,
            "action_loss": float(output["loss"].detach()),
            "retention_loss": float(retention_loss.detach()),
            "semantic_loss": float(semantic_loss.detach()),
            "total_loss": float(total_loss.detach()),
            "gradient_norm": float(grad_norm),
            "temperature": args.temperature,
            "retention": True,
            "semantic": True,
            "containment": False,
            "finite": True,
            "episode_id": batch["episode_ids"][0],
            "timestep": int(batch["timesteps"][0]),
        }
        records.append(record)
        print(json.dumps(record), flush=True)

    args.output.mkdir(parents=True)
    projector_state = {key: value.detach().cpu() for key, value in projector.state_dict().items()}
    torch.save(
        {
            "projector": projector_state,
            "head": {key: value.detach().cpu() for key, value in head.state_dict().items()},
            "proprio": {key: value.detach().cpu() for key, value in proprio.state_dict().items()},
            "optimizer": optimizer.state_dict(),
        },
        args.output / "modules.pt",
    )
    report = {
        "status": "passed",
        "experiment": "B3-retention-plus-semantic-no-containment",
        "steps": args.steps,
        "seed": args.seed,
        "losses": {
            "lambda_retention": args.lambda_retention,
            "lambda_semantic": args.lambda_semantic,
            "lambda_containment": 0.0,
            "semantic_temperature": args.temperature,
        },
        "matrix": {"retention": True, "semantic": True, "containment": False},
        "model_revision": source["revision"],
        "train_manifest_sha256": sha(args.train_manifest),
        "eval_manifest_sha256": sha(args.eval_manifest),
        "statistics_sha256": sha(args.statistics),
        "semantic_manifest_sha256": semantic_sha,
        "radio_manifest_sha256": radio_sha,
        "student_signal": "action-loss gradient times visual-token activation",
        "semantic_teacher": "SD1.5 null-text reconstruction",
        "retention_teacher": "nvidia/C-RADIOv3-L",
        "teacher_parameters_frozen": True,
        "records": records,
        "purpose": "engineering interface smoke; not benchmark performance",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
