#!/usr/bin/env python3
"""Build and validate the Goal18 Run Contract before any training."""
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
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--val-index", type=Path, required=True)
    parser.add_argument("--point-head", type=Path, required=True)
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
    train_rows = read_jsonl(args.train_index)
    val_rows = read_jsonl(args.val_index)
    eval_rows = read_jsonl(args.eval_sidecar)
    eval_summary = json.loads(args.eval_summary.read_text())
    split = json.loads(args.split_audit.read_text())

    train_images = {str(row["image_id"]) for row in train_rows}
    val_images = {str(row["image_id"]) for row in val_rows}
    eval_inits = {str(row["shared_init_id"]) for row in eval_rows}
    eval_image_names = {Path(str(row["image_path"])).name for row in eval_rows} | {
        Path(str(row["wrist_image_path"])).name for row in eval_rows
    }
    train_names = {Path(str(row.get("image_id", ""))).name for row in train_rows}
    eval_ids_in_train = {
        init_id
        for init_id in eval_inits
        if any(init_id in str(row.get("row_id", "")) for row in train_rows)
    }
    train_prompts = {str(row.get("prompt_id")) for row in train_rows}
    val_prompts = {str(row.get("prompt_id")) for row in val_rows}
    expected_prompts = set(config["prompts"])

    checks: dict[str, Any] = {
        "config_frozen": config.get("status") == "frozen_before_measurement",
        "training_scope_allowed": config.get("training_scope")
        in {"destination_offset_and_residual", "destination_offset", "destination_residual"},
        "oracle_fallback_disallowed": config.get("oracle_fallback_allowed") is False,
        "train_index_sha256": sha256_file(args.train_index)
        == config.get("train_index_sha256"),
        "val_index_sha256": sha256_file(args.val_index)
        == config.get("val_index_sha256"),
        "point_head_sha256": sha256_file(args.point_head)
        == config.get("point_head_sha256"),
        "train_index_count": len(train_rows) == 764 * 4,
        "val_index_count": len(val_rows) == 118 * 4,
        "train_image_count": len(train_images) == 764,
        "val_image_count": len(val_images) == 118,
        "train_prompt_coverage": train_prompts == expected_prompts,
        "val_prompt_coverage": val_prompts == expected_prompts,
        "train_split_only": all(str(row.get("split")) == "train" for row in train_rows),
        "val_split_only": all(str(row.get("split")) == "validation" for row in val_rows),
        "eval_init_count": len(eval_inits) == int(config.get("eval_init_count", 15)),
        "eval_sidecar_rows": len(eval_rows) == 60,
        "eval_sidecar_is_same_image": eval_summary.get("same_image_counterfactual") is True,
        "eval_not_in_train_ids": not eval_ids_in_train,
        "eval_image_names_not_in_train": not (eval_image_names & train_names),
        "split_audit_status_pass": split.get("status") == "PASS",
        "white_right_train_zero": split.get("gates", {}).get("white_right_train_zero") is True,
        "split_p0_disjoint_train": split.get("gates", {}).get("p0_eval_disjoint_from_train") is True,
        "split_p0_disjoint_validation": split.get("gates", {}).get("p0_eval_disjoint_from_validation") is True,
    }
    status = "PASS" if all(value is True for value in checks.values()) else "INVALID"
    payload = {
        "schema": "goal18-run-contract-v1",
        "status": status,
        "checks": checks,
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "train_index": str(args.train_index),
        "train_index_sha256": sha256_file(args.train_index),
        "val_index": str(args.val_index),
        "val_index_sha256": sha256_file(args.val_index),
        "point_head": str(args.point_head),
        "point_head_sha256": sha256_file(args.point_head),
        "eval_sidecar": str(args.eval_sidecar),
        "eval_sidecar_sha256": sha256_file(args.eval_sidecar),
        "eval_summary": str(args.eval_summary),
        "split_audit": str(args.split_audit),
        "counts": {
            "train_images": len(train_images),
            "val_images": len(val_images),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "eval_inits": len(eval_inits),
            "eval_rows": len(eval_rows),
        },
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "training": False,
        "notes": [
            "Goal18 only trains destination offset/residual correction; no full action head.",
            "The 15 eval inits and validation frames never enter training.",
            "The point head is frozen from Goal17 G4-MARGIN.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
