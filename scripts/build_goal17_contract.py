#!/usr/bin/env python3
"""Build the Goal17 Run Contract before any feature extraction or training."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_k4_lite_t4b_v8_release_settle import sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--train-index", type=Path, default=None)
    parser.add_argument("--val-index", type=Path, default=None)
    parser.add_argument("--eval-sidecar", type=Path, required=True)
    parser.add_argument("--eval-summary", type=Path, required=True)
    parser.add_argument("--split-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty jsonl: {path}")
    return rows


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text())
    train_manifest = read_jsonl(args.train_manifest)
    val_manifest = read_jsonl(args.val_manifest)
    source_train = Path(str(config["source_train_manifest"]))
    source_val = Path(str(config["source_val_manifest"]))
    train_index = read_jsonl(args.train_index) if args.train_index is not None else []
    val_index = read_jsonl(args.val_index) if args.val_index is not None else []
    eval_rows = read_jsonl(args.eval_sidecar)
    eval_summary = json.loads(args.eval_summary.read_text())
    split = json.loads(args.split_audit.read_text())

    train_images = {str(row["image_id"]) for row in train_index}
    val_images = {str(row["image_id"]) for row in val_index}
    eval_inits = {str(row["shared_init_id"]) for row in eval_rows}
    eval_image_names = {
        Path(str(row["image_path"])).name for row in eval_rows
    } | {Path(str(row["wrist_image_path"])).name for row in eval_rows}
    train_names = {Path(str(row.get("image_id", ""))).name for row in train_index}
    eval_ids_in_train = {
        init_id
        for init_id in eval_inits
        if any(init_id in str(row.get("row_id", "")) for row in train_index)
    }

    checks: dict[str, Any] = {
        "config_frozen": config.get("status") == "frozen_before_measurement",
        "training_scope_grounding_only": config.get("training_scope")
        == "grounding_point_head_only",
        "oracle_fallback_disallowed": config.get("oracle_fallback_allowed") is False,
        "gpu_is_1": str(os.environ.get("CUDA_VISIBLE_DEVICES", "")) == "1",
        "source_train_manifest_sha256": sha256_file(source_train)
        == config.get("source_train_manifest_sha256"),
        "source_val_manifest_sha256": sha256_file(source_val)
        == config.get("source_val_manifest_sha256"),
        "train_manifest_count": len(train_manifest) == 764 * 4,
        "val_manifest_count": len(val_manifest) == 118 * 4,
        "eval_init_count": len(eval_inits) == 15,
        "eval_sidecar_rows": len(eval_rows) == 60,
        "eval_sidecar_is_same_image": eval_summary.get("same_image_counterfactual") is True,
        "eval_not_in_train_ids": not eval_ids_in_train,
        "eval_image_names_not_in_train": not (eval_image_names & train_names),
        "split_audit_status_pass": split.get("status") == "PASS",
        "white_right_train_zero": split.get("gates", {}).get("white_right_train_zero") is True,
        "split_p0_disjoint_train": split.get("gates", {}).get("p0_eval_disjoint_from_train") is True,
        "split_p0_disjoint_validation": split.get("gates", {}).get("p0_eval_disjoint_from_validation") is True,
    }
    if args.train_index is not None and args.val_index is not None:
        checks.update(
            {
                "train_index_count": len(train_index) == 764 * 4,
                "val_index_count": len(val_index) == 118 * 4,
                "train_image_count": len(train_images) == 764,
                "val_image_count": len(val_images) == 118,
                "train_split_only": all(
                    str(row.get("split")) == "train" for row in train_index
                ),
                "val_split_only": all(
                    str(row.get("split")) == "validation" for row in val_index
                ),
            }
        )
    status = "PASS" if all(value is True for value in checks.values()) else "INVALID"
    payload = {
        "schema": "goal17-run-contract-v1",
        "status": status,
        "checks": checks,
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "train_manifest": str(args.train_manifest),
        "train_manifest_sha256": sha256_file(args.train_manifest),
        "val_manifest": str(args.val_manifest),
        "val_manifest_sha256": sha256_file(args.val_manifest),
        "source_train_manifest": str(source_train),
        "source_train_manifest_sha256": sha256_file(source_train),
        "source_val_manifest": str(source_val),
        "source_val_manifest_sha256": sha256_file(source_val),
        "train_index": str(args.train_index) if args.train_index is not None else None,
        "train_index_sha256": sha256_file(args.train_index)
        if args.train_index is not None
        else None,
        "val_index": str(args.val_index) if args.val_index is not None else None,
        "val_index_sha256": sha256_file(args.val_index)
        if args.val_index is not None
        else None,
        "eval_sidecar": str(args.eval_sidecar),
        "eval_sidecar_sha256": sha256_file(args.eval_sidecar),
        "eval_summary": str(args.eval_summary),
        "eval_summary_sha256": sha256_file(args.eval_summary),
        "split_audit": str(args.split_audit),
        "split_audit_sha256": sha256_file(args.split_audit),
        "counts": {
            "train_images": len(train_images),
            "val_images": len(val_images),
            "train_rows": len(train_index),
            "val_rows": len(val_index),
            "eval_inits": len(eval_inits),
            "eval_rows": len(eval_rows),
        },
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "training": False,
        "index_status": "complete"
        if args.train_index is not None and args.val_index is not None
        else "pending",
        "notes": [
            "This contract is written before the 4-prompt head is trained.",
            "The 15 eval inits and validation frames never enter the train index.",
            "Goal17 only trains the point head; no residual/route/action head is trained.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
