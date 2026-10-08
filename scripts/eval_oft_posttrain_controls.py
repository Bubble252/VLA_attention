"""Evaluate post-training action evidence under semantic-map controls.

This is a mechanism evaluation only.  It restores frozen B0--B4 checkpoints,
computes the native action-loss gradient x activation map on held-out
observation/action chunks, and compares its support mass under correct,
same-image role-swapped (operational wrong-word), wrong-image, random and
area-matched random semantic maps.  It never trains, rolls out, or reports
LIBERO success.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

GROUPS = ("B0", "B1", "B2", "B3", "B4")
CAMERAS = ("image", "wrist_image")
ROLES = ("source", "target")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--eval-manifest", type=Path, required=True)
    p.add_argument("--statistics", type=Path, required=True)
    p.add_argument("--semantic-root", type=Path, required=True)
    p.add_argument("--semantic-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--groups", default=",".join(GROUPS))
    p.add_argument("--checkpoint-step", type=int, default=10080)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-transitions", type=int, default=0)
    return p.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def normalize_np(values):
    import numpy as np

    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = float(values.sum())
    return values / max(total, 1e-12)


def support_mass(action, support):
    """Return mass on a [0, 1] semantic support map."""
    import numpy as np

    a = normalize_np(action)
    s = np.asarray(support, dtype=np.float64)
    s = np.maximum(s, 0.0)
    s = s / max(float(s.max()), 1e-12)
    return float((a * np.clip(s, 0.0, 1.0)).sum())


def kl_to_support(action, support):
    import numpy as np

    a = normalize_np(action)
    s = normalize_np(support)
    return float((s * (np.log(s + 1e-8) - np.log(a + 1e-8))).sum())


def load_map(root: Path, row: dict[str, Any]):
    import numpy as np

    path = root / f"{row['sample_id']}.npy"
    value = np.load(path, allow_pickle=False).astype("float32")
    if value.shape != (16, 16) or not np.isfinite(value).all() or (value < 0).any():
        raise ValueError(f"invalid semantic map: {path}")
    return value.reshape(-1)


class SemanticIndex:
    def __init__(self, manifest: Path, root: Path):
        self.root = root
        rows = read_rows(manifest)
        self.by_key = {
            (r["episode_id"], int(r["timestep"]), r["camera"], r["role"]): r
            for r in rows
        }
        if len(self.by_key) != len(rows):
            raise ValueError("duplicate semantic cache key")
        self.by_episode = {}
        for r in rows:
            self.by_episode.setdefault((r["episode_id"], int(r["timestep"])), r)
        self.keys = list(self.by_episode)

    def _pair(self, episode: str, timestep: int, camera: str):
        out = {}
        for role in ROLES:
            row = self.by_key.get((episode, timestep, camera, role))
            if row is None:
                raise KeyError(f"missing semantic map: {episode}/{timestep}/{camera}/{role}")
            out[role] = load_map(self.root, row)
        return out

    def support(self, episode: str, timestep: int, control: str, sample_index: int):
        import numpy as np

        pairs = [self._pair(episode, timestep, camera) for camera in CAMERAS]
        if control == "correct":
            return np.concatenate(
                [np.maximum(pair["source"], pair["target"]) for pair in pairs]
            )
        if control == "role_swapped_same_image":
            # The operational wrong-word control uses the opposite phrase role
            # from the same observation.  This preserves pixels and camera.
            return np.concatenate([pair["source"] for pair in pairs])
        if control == "wrong_image":
            peer_episode, peer_timestep = self.keys[(sample_index + 1) % len(self.keys)]
            if peer_episode == episode:
                peer_episode, peer_timestep = self.keys[(sample_index + 2) % len(self.keys)]
            peer = [self._pair(peer_episode, peer_timestep, camera) for camera in CAMERAS]
            return np.concatenate(
                [np.maximum(pair["source"], pair["target"]) for pair in peer]
            )
        if control == "random":
            rng = np.random.default_rng(17000 + sample_index)
            return rng.random(512, dtype=np.float32)
        if control == "area_matched_random":
            rng = np.random.default_rng(27000 + sample_index)
            correct = np.concatenate(
                [np.maximum(pair["source"], pair["target"]) for pair in pairs]
            )
            k = max(1, int(np.count_nonzero(correct >= np.quantile(correct, 0.8))))
            out = np.zeros(512, dtype=np.float32)
            out[rng.choice(512, size=min(k, 512), replace=False)] = rng.random(
                min(k, 512)
            )
            return out
        raise ValueError(f"unknown control: {control}")


def restore_group(args, group: str):
    import torch
    from peft import PeftModel
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector

    run_dir = args.results_root / f"{group}_seed{args.seed}"
    checkpoint = run_dir / f"checkpoint_{args.checkpoint_step:05d}"
    required = [
        run_dir / "final.json",
        checkpoint / "adapter",
        checkpoint / "modules.pt",
        checkpoint / "checkpoint.json",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if json.loads((run_dir / "final.json").read_text()).get("status") != "passed":
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
    # A visual activation gradient needs the vision graph to remain
    # differentiable.  autograd.grad targets the activation only, so parameter
    # gradients are not accumulated.
    for parameter in policy.base_model.model.vision_backbone.parameters():
        parameter.requires_grad_(True)
    head = L1RegressionActionHead(input_dim=base.llm_dim, hidden_dim=base.llm_dim).cuda().bfloat16()
    proprio = ProprioProjector(llm_dim=base.llm_dim, proprio_dim=8).cuda().bfloat16()
    state = torch.load(checkpoint / "modules.pt", map_location="cpu", weights_only=False)
    head.load_state_dict(state["head"])
    proprio.load_state_dict(state["proprio"])
    policy.eval(); head.eval(); proprio.eval()
    return policy, head, proprio, base.vision_backbone.get_num_patches() * 2, checkpoint


def evaluate_group(args, group: str, dataset, semantic: SemanticIndex, patch_count: int):
    import numpy as np
    import torch
    from vla_attention.adapters.oft_forward import forward_l1

    policy, head, proprio, patch_count, checkpoint = restore_group(args, group)
    limit = len(dataset) if args.max_transitions <= 0 else min(args.max_transitions, len(dataset))
    controls = ("correct", "role_swapped_same_image", "wrong_image", "random", "area_matched_random")
    records = []
    started = time.time()
    for index in range(limit):
        batch = dataset.collate([dataset[index]])
        captured = []

        def capture(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(value):
                value.retain_grad()
                captured[:] = [value]
            return output

        hook = policy.base_model.model.vision_backbone.register_forward_hook(capture)
        with torch.enable_grad():
            output = forward_l1(
                policy, head, proprio, batch, device="cuda", visual_token_count=patch_count
            )
            if not captured or not captured[0].requires_grad:
                hook.remove()
                raise RuntimeError("visual activation is not differentiable")
            (gradient,) = torch.autograd.grad(
                output["loss"], captured[0], retain_graph=False, create_graph=False
            )
        hook.remove()
        features = captured[0][:, :patch_count]
        action_map = (
            gradient[:, :patch_count].float().mul(features.float()).sum(dim=-1).abs()
        )[0].detach().cpu().numpy()
        episode, timestep = batch["episode_ids"][0], int(batch["timesteps"][0])
        row = {
            "index": index,
            "episode_id": episode,
            "timestep": timestep,
            "action_loss": float(output["loss"].detach().cpu()),
            "action_map_shape": list(action_map.shape),
            "gradient_finite": bool(np.isfinite(action_map).all()),
        }
        for control in controls:
            support = semantic.support(episode, timestep, control, index)
            row[f"{control}_support_mass"] = support_mass(action_map, support)
            row[f"{control}_support_kl"] = kl_to_support(action_map, support)
        records.append(row)
        if (index + 1) % 100 == 0:
            print(json.dumps({"group": group, "processed": index + 1, "total": limit}), flush=True)
    metrics = {"group": group, "n_transitions": len(records)}
    for control in controls:
        values = [r[f"{control}_support_mass"] for r in records]
        kls = [r[f"{control}_support_kl"] for r in records]
        metrics[f"{control}_support_mass_mean"] = float(np.mean(values))
        metrics[f"{control}_support_mass_std"] = float(np.std(values))
        metrics[f"{control}_support_kl_mean"] = float(np.mean(kls))
        metrics[f"{control}_support_kl_std"] = float(np.std(kls))
    metrics.update(
        {
            "status": "passed",
            "checkpoint": str(checkpoint),
            "checkpoint_modules_sha256": sha(checkpoint / "modules.pt"),
            "eval_manifest_sha256": sha(args.eval_manifest),
            "semantic_manifest_sha256": sha(args.semantic_manifest),
            "metric_source": "action_loss_gradient_times_activation",
            "raw_attention_proxy": {
                "available": False,
                "reason": "pinned OpenVLA/OFT evaluator does not expose a comparable vision attention map",
            },
            "no_rollout_success_claim": True,
            "elapsed_seconds": time.time() - started,
        }
    )
    group_dir = args.output / group
    group_dir.mkdir(parents=True, exist_ok=True)
    (group_dir / "records.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    )
    (group_dir / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    del policy, head, proprio
    gc.collect(); torch.cuda.empty_cache()
    return metrics


def main():
    a = args()
    groups = tuple(x.strip() for x in a.groups.split(",") if x.strip())
    if set(groups) - set(GROUPS):
        raise ValueError(f"unknown groups: {sorted(set(groups) - set(GROUPS))}")
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)

    import numpy as np
    import torch
    from vla_attention.benchmarks.oft_rlds import EpisodeDataset

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    semantic = SemanticIndex(a.semantic_manifest, a.semantic_root)
    dataset = EpisodeDataset(a.eval_manifest, a.statistics, a.model)
    if a.max_transitions <= 0 and len(dataset) != len(semantic.keys):
        # Each episode/timestep has one H=8 transition row and four semantic
        # rows; a mismatch would make wrong-image pairing non-reproducible.
        raise ValueError(f"eval transition/semantic key mismatch: {len(dataset)} vs {len(semantic.keys)}")
    # Probe the model once to obtain its native visual token count.
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    config = OpenVLAConfig.from_pretrained(a.model, local_files_only=True)
    probe = OpenVLAForActionPrediction.from_pretrained(
        a.model, config=config, local_files_only=True, low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda()
    probe.vision_backbone.set_num_images_in_input(2)
    patch_count = probe.vision_backbone.get_num_patches() * 2
    del probe; gc.collect(); torch.cuda.empty_cache()
    summaries = [evaluate_group(a, group, dataset, semantic, patch_count) for group in groups]
    report = {
        "status": "passed",
        "experiment": "P6 B0-B4 held-out action evidence controls",
        "groups": summaries,
        "controls": [
            "correct",
            "role_swapped_same_image (operational wrong-word)",
            "wrong_image",
            "random",
            "area_matched_random",
        ],
        "eval_manifest_sha256": sha(a.eval_manifest),
        "semantic_manifest_sha256": sha(a.semantic_manifest),
        "no_rollout_success_claim": True,
        "raw_attention_proxy_available": False,
        "created_at_unix": time.time(),
    }
    (a.output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
