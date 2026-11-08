#!/usr/bin/env python3
"""Train a PointQueryHead with same-image counterfactual margin losses.

This is Goal17's grounding-only trainer.  It never trains a policy, residual,
route head, or action head.  The model consumes frozen Qwen visual/language
hidden states and predicts object/destination image points.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_c1_point_head import (
    PointFeatureDataset,
    _move_batch as _move_point_batch,
    collate_point_batch,
    cosine_with_warmup,
    evaluate_point_head,
    point_l1_loss,
)
from starVLA.model.modules.point_head import PointQueryHead


DESTINATION_PAIRS = (("red_left", "red_right"), ("white_left", "white_right"))
OBJECT_PAIRS = (("red_left", "white_left"), ("red_right", "white_right"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--val-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts", default="red_left,red_right,white_left,white_right")
    parser.add_argument("--val-prompts", default="")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--group-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=3584)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--margin-dest", type=float, default=0.30)
    parser.add_argument("--margin-obj", type=float, default=0.10)
    parser.add_argument("--lambda-dest", type=float, default=0.5)
    parser.add_argument("--lambda-obj", type=float, default=0.5)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--limit-groups", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


class CounterfactualGroupDataset(Dataset):
    """One item is all prompt variants that share an image."""

    def __init__(self, index_path: Path, prompts: tuple[str, ...], limit: int = 0) -> None:
        self.base = PointFeatureDataset(index_path, limit=0)
        self.prompts = tuple(prompts)
        if len(set(self.prompts)) != len(self.prompts):
            raise ValueError("prompts must be unique")
        by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.base.rows:
            if str(row["prompt_id"]) not in self.prompts:
                continue
            by_image[str(row["image_id"])].append(row)
        groups: list[list[dict[str, Any]]] = []
        for image_id, rows in sorted(by_image.items()):
            prompt_map = {str(row["prompt_id"]): row for row in rows}
            if set(prompt_map) != set(self.prompts):
                raise ValueError(
                    f"{image_id}: expected prompts {self.prompts}, got {sorted(prompt_map)}"
                )
            groups.append([prompt_map[prompt] for prompt in self.prompts])
        if limit > 0:
            groups = groups[:limit]
        if not groups:
            raise ValueError(f"no counterfactual groups in {index_path}")
        self.groups = groups
        self.row_to_index = {row["row_id"]: i for i, row in enumerate(self.base.rows)}

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> list[dict[str, Any]]:
        return [self.base[self.row_to_index[row["row_id"]]] for row in self.groups[index]]


def collate_group_batch(groups: list[list[dict[str, Any]]]) -> dict[str, Any]:
    if not groups or any(len(group) != len(groups[0]) for group in groups):
        raise ValueError("all counterfactual groups must have the same prompt count")
    flat = [item for group in groups for item in group]
    batch = collate_point_batch(flat)
    batch["group_ids"] = torch.tensor(
        [group_index for group_index, group in enumerate(groups) for _ in group],
        dtype=torch.long,
    )
    return batch


def _move_group_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = _move_point_batch(batch, device)
    moved["group_ids"] = batch["group_ids"].to(device)
    return moved


def _pair_distance(
    predicted: torch.Tensor,
    *,
    index_a: int,
    index_b: int,
    query: int,
) -> torch.Tensor:
    return torch.linalg.vector_norm(predicted[index_a, query] - predicted[index_b, query])


def counterfactual_margin_losses(
    predicted: torch.Tensor,
    batch: dict[str, Any],
    *,
    margin_dest: float,
    margin_obj: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return destination/object hinge losses and diagnostic distances."""
    group_ids = batch["group_ids"]
    prompt_ids = batch["metadata"]["prompt_id"]
    destination_losses: list[torch.Tensor] = []
    object_losses: list[torch.Tensor] = []
    destination_distances: list[float] = []
    object_distances: list[float] = []

    for group_index in range(int(group_ids.max().item()) + 1):
        indices = (group_ids == group_index).nonzero(as_tuple=False).flatten().tolist()
        by_prompt = {str(prompt_ids[index]): int(index) for index in indices}
        for prompt_a, prompt_b in DESTINATION_PAIRS:
            if prompt_a not in by_prompt or prompt_b not in by_prompt:
                continue
            distance = _pair_distance(
                predicted,
                index_a=by_prompt[prompt_a],
                index_b=by_prompt[prompt_b],
                query=1,
            )
            destination_losses.append(F.relu(float(margin_dest) - distance).pow(2))
            destination_distances.append(float(distance.detach().cpu()))
        for prompt_a, prompt_b in OBJECT_PAIRS:
            if prompt_a not in by_prompt or prompt_b not in by_prompt:
                continue
            distance = _pair_distance(
                predicted,
                index_a=by_prompt[prompt_a],
                index_b=by_prompt[prompt_b],
                query=0,
            )
            object_losses.append(F.relu(float(margin_obj) - distance).pow(2))
            object_distances.append(float(distance.detach().cpu()))

    zero = predicted.sum() * 0.0
    dest_loss = torch.stack(destination_losses).mean() if destination_losses else zero
    obj_loss = torch.stack(object_losses).mean() if object_losses else zero
    stats = {
        "destination_pair_count": len(destination_distances),
        "destination_distance_mean": float(np.mean(destination_distances))
        if destination_distances
        else 0.0,
        "object_pair_count": len(object_distances),
        "object_distance_mean": float(np.mean(object_distances)) if object_distances else 0.0,
        "destination_margin_loss": float(dest_loss.detach().cpu()),
        "object_margin_loss": float(obj_loss.detach().cpu()),
    }
    return dest_loss, obj_loss, stats


def main() -> int:
    args = parse_args()
    prompts = tuple(value.strip() for value in args.prompts.split(",") if value.strip())
    val_prompts = tuple(
        value.strip() for value in (args.val_prompts or args.prompts).split(",") if value.strip()
    )
    if not prompts:
        raise ValueError("--prompts cannot be empty")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.steps < 1 or args.group_batch_size < 1:
        raise ValueError("steps and group-batch-size must be positive")
    if args.margin_dest <= 0 or args.margin_obj <= 0:
        raise ValueError("margins must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    train_dataset = CounterfactualGroupDataset(
        args.train_index, prompts, limit=args.limit_groups
    )
    val_dataset = PointFeatureDataset(args.val_index, limit=args.limit_val)
    val_dataset.rows = [
        row for row in val_dataset.rows if str(row["prompt_id"]) in val_prompts
    ]
    if not val_dataset.rows:
        raise ValueError("validation prompt filter produced no rows")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.group_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_group_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_point_batch,
    )
    model = PointQueryHead(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_with_warmup(step, args.steps, args.warmup)
    )
    args.output.mkdir(parents=True)
    best_state = None
    best_metrics = None
    history: list[dict[str, Any]] = []
    iterator = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = _move_group_batch(batch, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            predicted, _ = model(
                batch["visual_tokens"],
                batch["visual_mask"],
                batch["language_tokens"],
                batch["language_mask"],
                visual_view_ids=batch["visual_view_ids"],
            )
            l1_loss = point_l1_loss(predicted.float(), batch["points"].float())
            dest_loss, obj_loss, margin_stats = counterfactual_margin_losses(
                predicted.float(),
                batch,
                margin_dest=args.margin_dest,
                margin_obj=args.margin_obj,
            )
            loss = (
                l1_loss
                + float(args.lambda_dest) * dest_loss
                + float(args.lambda_obj) * obj_loss
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        if step % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": float(loss.detach().cpu()),
                        "l1_loss": float(l1_loss.detach().cpu()),
                        "lr": float(scheduler.get_last_lr()[0]),
                        **margin_stats,
                    }
                ),
                flush=True,
            )
        if step % args.val_every == 0 or step == args.steps:
            metrics = evaluate_point_head(model, val_loader, device)
            metrics.update({"step": step, "train_loss": float(loss.detach().cpu())})
            history.append(metrics)
            if best_metrics is None or metrics["point_l1_px"]["all"] < best_metrics["point_l1_px"]["all"]:
                best_metrics = metrics
                best_state = {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                }
            print(json.dumps({"eval": metrics}, ensure_ascii=False), flush=True)

    final_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    final_metrics = history[-1] if history else None
    checkpoint_path = args.output / "point_head.pt"
    torch.save(
        {
            "model": final_state,
            "final_model": final_state,
            "best_model": best_state,
            "final_metrics": final_metrics,
            "best_metrics": best_metrics,
            "args": vars(args),
            "history": history,
        },
        checkpoint_path,
    )
    summary = {
        "schema": "goal17-point-head-training-v1",
        "status": "complete",
        "train_rows": len(train_dataset.base.rows),
        "train_groups": len(train_dataset),
        "val_rows": len(val_dataset),
        "prompts": list(prompts),
        "steps": args.steps,
        "group_batch_size": args.group_batch_size,
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "gate_checkpoint": "final_model",
        "margin": {
            "destination": args.margin_dest,
            "object": args.margin_obj,
            "lambda_destination": args.lambda_dest,
            "lambda_object": args.lambda_obj,
        },
        "final_metrics": final_metrics,
        "best_metrics": best_metrics,
        "history": history,
        "notes": [
            "Only PointQueryHead parameters are trained; Qwen and the P4 adapter are frozen.",
            "Margin pairs are same-image prompt interventions, not simulator-success labels.",
            "The final checkpoint, not best-on-validation, is used for the Goal17 Gate.",
        ],
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
