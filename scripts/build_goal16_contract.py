#!/usr/bin/env python3
"""Build and validate the Goal16 Run Contract before any arm starts."""
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

from scripts.audit_k4_lite_t4b_v8_release_settle import sha256_file
from scripts.eval_oft_single_checkpoint import checkpoint_fingerprint, resolve_checkpoint

EXPECTED_CONFIG_SHA256 = "34731e83e56d6c2ca075bdaecbbbb87671379053194b28a4561c06271ea8e502"
EXPECTED_MANIFEST_SHA256 = "e9564ca9aef800e618e3fdb5d144892eabe9471815bbd24bae3140583946f45e"
EXPECTED_CHECKPOINT_FINGERPRINT = "747a28690383ca45e795518121e9aed3f4610265c14618a69f203922b4238e20"
EXPECTED_STATISTICS_SHA256 = "6d5ebb47bc5132df581aae0481d529d04a0821c331859c05a9818d78156c9538"
EXPECTED_POINT_HEAD_SHA256 = "693f0dd8086a4d64cd4f1064a7f4d5c1f934f0914123df5478cdc1f86815fce7"
EXPECTED_SIDECAR_SHA256 = "7fb06ba005b9cc82453c2bab98ac8444d5fe313e7c4e094413dd566b91efe43a"
CALIBRATION_INIT_IDS = {
    "init_0e94a4e6b9622c6b",
    "init_2edf49999eb35566",
    "init_3500df19a4d3b5a1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--point-head", type=Path, required=True)
    parser.add_argument("--grounding-sidecar", type=Path, required=True)
    parser.add_argument("--split-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text())
    manifest_rows = [
        json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()
    ]
    selected = [row for row in manifest_rows if row.get("prompt_id") == "white_right"]
    selected.sort(key=lambda row: str(row["shared_init_id"]))
    init_ids = {str(row["shared_init_id"]) for row in selected}
    init_shas = {str(row["init_state_sha256"]) for row in selected}

    split = json.loads(args.split_audit.read_text())
    grounding_summary = json.loads((args.grounding_sidecar.parent / "summary.json").read_text())
    sidecar_sha = sha256_file(args.grounding_sidecar)

    layout = resolve_checkpoint(args.checkpoint)
    fingerprint = checkpoint_fingerprint(layout["fingerprint_files"])
    checks: dict[str, Any] = {
        "config_sha256": sha256_file(args.config) == EXPECTED_CONFIG_SHA256,
        "manifest_sha256": sha256_file(args.manifest) == EXPECTED_MANIFEST_SHA256,
        "checkpoint_fingerprint": fingerprint == EXPECTED_CHECKPOINT_FINGERPRINT,
        "statistics_sha256": sha256_file(args.statistics) == EXPECTED_STATISTICS_SHA256,
        "point_head_sha256": sha256_file(args.point_head) == EXPECTED_POINT_HEAD_SHA256,
        "grounding_sidecar_sha256": sidecar_sha == EXPECTED_SIDECAR_SHA256,
        "grounding_sidecar_complete": grounding_summary.get("selected_init_count") == 15,
        "grounding_sidecar_matches_manifest": {
            str(row["shared_init_id"]) for row in (
                json.loads(line)
                for line in args.grounding_sidecar.read_text().splitlines()
                if line.strip()
            )
        } == init_ids,
        "split_audit_status_pass": split.get("status") == "PASS",
        "white_right_train_zero": split.get("gates", {}).get("white_right_train_zero") is True,
        "split_p0_disjoint_train": split.get("gates", {}).get("p0_eval_disjoint_from_train") is True,
        "split_p0_disjoint_validation": split.get("gates", {}).get("p0_eval_disjoint_from_validation") is True,
        "split_p0_matches_rollout": split.get("gates", {}).get("p0_eval_matches_rollout_source") is True,
        "split_init_set_matches_manifest": {
            str(value) for value in split.get("p0_white_right", {}).get("init_state_shas", [])
        } == init_shas,
        "init_count_15": len(selected) == 15,
        "calibration_count_3": sum(
            str(row["shared_init_id"]) in CALIBRATION_INIT_IDS for row in selected
        ) == 3,
        "confirmation_count_12": sum(
            str(row["shared_init_id"]) not in CALIBRATION_INIT_IDS for row in selected
        ) == 12,
        "config_frozen": config.get("status") == "frozen_before_measurement",
        "no_training": True,
        "gpu_is_1": str(os.environ.get("CUDA_VISIBLE_DEVICES", "")) == "1",
        "point_head_sha256_matches_config": (
            config.get("grounding", {}).get("point_head_sha256")
            == EXPECTED_POINT_HEAD_SHA256
        ),
        "oracle_fallback_disallowed": (
            config.get("grounding", {}).get("oracle_fallback_allowed") is False
        ),
    }
    # R-PRED is expected to be INVALID here, but the contract is about the
    # experiment setup, not about pretending the predictor passed.
    checks["grounding_predictor_status"] = (
        "PASS" if grounding_summary.get("passed") is True else "INVALID"
    )
    status = "PASS" if all(
        value is True for key, value in checks.items() if key != "grounding_predictor_status"
    ) else "INVALID"
    payload = {
        "schema": "goal16-run-contract-v1",
        "status": status,
        "checks": checks,
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "checkpoint": str(args.checkpoint),
        "checkpoint_fingerprint": fingerprint,
        "statistics": str(args.statistics),
        "statistics_sha256": sha256_file(args.statistics),
        "point_head": str(args.point_head),
        "point_head_sha256": sha256_file(args.point_head),
        "grounding_sidecar": str(args.grounding_sidecar),
        "grounding_sidecar_sha256": sidecar_sha,
        "grounding_summary": str(args.grounding_sidecar.parent / "summary.json"),
        "grounding_status": checks["grounding_predictor_status"],
        "split_audit": str(args.split_audit),
        "split_audit_sha256": sha256_file(args.split_audit),
        "init_count": len(selected),
        "calibration_init_ids": sorted(CALIBRATION_INIT_IDS),
        "confirmation_init_count": sum(
            str(row["shared_init_id"]) not in CALIBRATION_INIT_IDS for row in selected
        ),
        "seed": 7,
        "arms": ["B0", "R-GEO", "R-NAIVE", "R-PRED", "R-SHUF"],
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "training": False,
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
