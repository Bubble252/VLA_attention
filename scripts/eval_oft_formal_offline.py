"""Evaluate trained OpenVLA/OFT B0--B4 checkpoints on held-out offline actions.

This is offline action prediction only. It intentionally refuses to report a
LIBERO rollout success rate. The evaluator restores the LoRA adapter, native
OFT action head and proprio projector saved by run_oft_b0_b4_formal_offline.py,
then computes native normalized-action metrics on an episode-isolated eval
manifest.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

GROUPS = ("B0", "B1", "B2", "B3", "B4")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", default=",".join(GROUPS))
    parser.add_argument("--checkpoint-step", type=int, default=10080)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-transitions", type=int, default=0)
    return parser.parse_args()


def metric_block(errors_abs, errors_sq, pred, target, native_l1_losses):
    import numpy as np

    if errors_sq.size == 0:
        raise ValueError("empty metric block")
    gripper_pred = pred[..., 6] >= 0
    gripper_target = target[..., 6] >= 0
    return {
        "n_transitions": int(pred.shape[0]),
        "n_action_values": int(pred.size),
        "action_mse": float(errors_sq.mean()),
        "action_mae": float(errors_abs.mean()),
        "native_l1_forward_loss_mean": float(np.mean(native_l1_losses)),
        "translation_mse": float(errors_sq[..., :3].mean()),
        "translation_mae": float(errors_abs[..., :3].mean()),
        "rotation_mse": float(errors_sq[..., 3:6].mean()),
        "rotation_mae": float(errors_abs[..., 3:6].mean()),
        "gripper_mse": float(errors_sq[..., 6].mean()),
        "gripper_mae": float(errors_abs[..., 6].mean()),
        "gripper_accuracy_sign_normalized": float((gripper_pred == gripper_target).mean()),
        "per_dim_mse": [float(x) for x in errors_sq.reshape(-1, 7).mean(axis=0)],
        "per_dim_mae": [float(x) for x in errors_abs.reshape(-1, 7).mean(axis=0)],
        "per_horizon_action_mse": [float(x) for x in errors_sq.mean(axis=(0, 2))],
        "per_horizon_action_mae": [float(x) for x in errors_abs.mean(axis=(0, 2))],
    }


def evaluate_group(args: argparse.Namespace, group: str, dataset, patch_count: int) -> dict[str, Any]:
    import numpy as np
    import torch
    from peft import PeftModel
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector
    from vla_attention.adapters.oft_forward import forward_l1

    run_dir = args.results_root / f"{group}_seed{args.seed}"
    checkpoint = run_dir / f"checkpoint_{args.checkpoint_step:05d}"
    for required in (run_dir / "final.json", checkpoint / "adapter", checkpoint / "modules.pt", checkpoint / "checkpoint.json"):
        if not required.exists():
            raise FileNotFoundError(required)
    final = read_json(run_dir / "final.json")
    if final.get("status") != "passed":
        raise ValueError(f"{group} final.json is not passed")

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
    policy = PeftModel.from_pretrained(base, checkpoint / "adapter", is_trainable=False)
    head = L1RegressionActionHead(input_dim=base.llm_dim, hidden_dim=base.llm_dim).cuda().bfloat16()
    proprio = ProprioProjector(llm_dim=base.llm_dim, proprio_dim=8).cuda().bfloat16()
    state = torch.load(checkpoint / "modules.pt", map_location="cpu", weights_only=False)
    head.load_state_dict(state["head"])
    proprio.load_state_dict(state["proprio"])
    policy.eval(); head.eval(); proprio.eval()

    limit = len(dataset) if args.max_transitions <= 0 else min(args.max_transitions, len(dataset))
    records = []
    by_episode: dict[str, list[int]] = defaultdict(list)
    predictions = []
    targets = []
    native_l1_losses = []
    started = time.time()
    with torch.no_grad():
        for index in range(limit):
            batch = dataset.collate([dataset[index]])
            output = forward_l1(policy, head, proprio, batch, device="cuda", visual_token_count=patch_count)
            pred = output["prediction"].float().cpu().numpy()[0]
            target = batch["actions"].float().cpu().numpy()[0]
            err = pred - target
            record = {
                "index": index,
                "group": group,
                "episode_id": batch["episode_ids"][0],
                "timestep": int(batch["timesteps"][0]),
                "action_mse": float(np.square(err).mean()),
                "action_mae": float(np.abs(err).mean()),
                "translation_mse": float(np.square(err[:, :3]).mean()),
                "rotation_mse": float(np.square(err[:, 3:6]).mean()),
                "gripper_mse": float(np.square(err[:, 6]).mean()),
                "gripper_accuracy_sign_normalized": float(((pred[:, 6] >= 0) == (target[:, 6] >= 0)).mean()),
                "native_l1_forward_loss": float(output["loss"].detach().cpu()),
            }
            by_episode[record["episode_id"]].append(len(records))
            records.append(record)
            predictions.append(pred)
            targets.append(target)
            native_l1_losses.append(record["native_l1_forward_loss"])
    pred_array = np.stack(predictions, axis=0)
    target_array = np.stack(targets, axis=0)
    abs_err = np.abs(pred_array - target_array)
    sq_err = np.square(pred_array - target_array)
    per_episode = {}
    for episode_id, indices in by_episode.items():
        idx = np.asarray(indices, dtype=np.int64)
        per_episode[episode_id] = metric_block(abs_err[idx], sq_err[idx], pred_array[idx], target_array[idx], [native_l1_losses[i] for i in indices])
    overall = metric_block(abs_err, sq_err, pred_array, target_array, native_l1_losses)
    overall.update({
        "status": "passed",
        "group": group,
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha(checkpoint / "modules.pt"),
        "final_json_sha256": sha(run_dir / "final.json"),
        "eval_manifest_sha256": sha(args.eval_manifest),
        "statistics_sha256": sha(args.statistics),
        "metric_source": "offline_prediction",
        "rollout_success_rate": None,
        "no_rollout_success_claim": True,
        "elapsed_seconds": time.time() - started,
    })
    group_dir = args.output / group
    group_dir.mkdir(parents=True, exist_ok=True)
    (group_dir / "records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    (group_dir / "per_episode.json").write_text(json.dumps(per_episode, indent=2, ensure_ascii=False) + "\n")
    (group_dir / "summary.json").write_text(json.dumps(overall, indent=2, ensure_ascii=False) + "\n")
    del policy, base, head, proprio, state
    gc.collect()
    torch.cuda.empty_cache()
    return overall


def main() -> None:
    args = parse_args()
    groups = tuple(x.strip() for x in args.groups.split(",") if x.strip())
    unknown = set(groups) - set(GROUPS)
    if unknown:
        raise ValueError(f"unknown groups: {sorted(unknown)}")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    import numpy as np
    import torch
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from vla_attention.benchmarks.oft_rlds import EpisodeDataset

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset = EpisodeDataset(args.eval_manifest, args.statistics, args.model)
    config = OpenVLAConfig.from_pretrained(args.model, local_files_only=True)
    probe = OpenVLAForActionPrediction.from_pretrained(
        args.model,
        config=config,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda()
    probe.vision_backbone.set_num_images_in_input(2)
    patch_count = probe.vision_backbone.get_num_patches() * 2
    del probe
    gc.collect(); torch.cuda.empty_cache()

    summaries = []
    for group in groups:
        summaries.append(evaluate_group(args, group, dataset, patch_count))

    by_group = {row["group"]: row for row in summaries}
    comparisons = {}
    def delta(candidate: str, baseline: str) -> dict[str, float]:
        metrics = ("action_mse", "action_mae", "translation_mse", "rotation_mse", "gripper_mse", "gripper_accuracy_sign_normalized")
        out = {}
        for metric in metrics:
            out[f"{candidate}_minus_{baseline}_{metric}"] = float(by_group[candidate][metric] - by_group[baseline][metric])
        return out
    for candidate, baseline in (("B1", "B0"), ("B2", "B0"), ("B3", "B1"), ("B4", "B3"), ("B4", "B2")):
        if candidate in by_group and baseline in by_group:
            comparisons[f"{candidate}_vs_{baseline}"] = delta(candidate, baseline)
    report = {
        "status": "passed",
        "experiment": "P6 VLA B0-B4 held-out offline action prediction",
        "metric_source": "offline_prediction",
        "no_rollout_success_claim": True,
        "groups": summaries,
        "comparisons": comparisons,
        "eval_manifest": str(args.eval_manifest),
        "eval_manifest_sha256": sha(args.eval_manifest),
        "statistics_sha256": sha(args.statistics),
        "result_root": str(args.results_root),
        "max_transitions": args.max_transitions or None,
        "created_at_unix": time.time(),
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
