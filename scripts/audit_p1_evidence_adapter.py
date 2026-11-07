#!/usr/bin/env python3
"""P1-A evidence-adapter qualification for route-centroid evidence.

This audit does not train a model.  It checks whether the frozen T1 route
centroid geometry is a valid deployment-time evidence adapter for P1:
finite, non-degenerate, split-disjoint, and free of the held-out
``white_right`` prompt in training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.k4_lite_t4_common import load_t4_cache  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def is_white_right(language: Any) -> bool:
    text = str(language).lower()
    return "white mug" in text and "right plate" in text


def summarize_geometry(path: Path) -> dict[str, Any]:
    cache = load_t4_cache(path)
    arrays = cache["arrays"]
    meta = cache["meta"]
    geometry = np.asarray(arrays["geometry"], dtype=np.float64)
    if geometry.ndim != 2 or geometry.shape[1] != 6:
        raise ValueError(f"{path}: geometry must be [N, 6], got {geometry.shape}")
    if len(meta) != geometry.shape[0]:
        raise ValueError(f"{path}: meta/geometry row mismatch")

    finite = np.isfinite(geometry).all(axis=1)
    delta_norm = np.linalg.norm(geometry[:, 4:6], axis=1)
    nonzero = finite & (delta_norm > 1e-6)
    in_grid = ((geometry[:, :4] >= -1.0 - 1e-6) & (geometry[:, :4] <= 1.0 + 1e-6)).all(axis=1)
    valid = finite & nonzero
    languages = Counter(str(row.get("language")) for row in meta)
    episodes = {str(row["episode_id"]) for row in meta}
    white_right = sum(is_white_right(row.get("language")) for row in meta)
    return {
        "path": str(path),
        "transitions": int(geometry.shape[0]),
        "episodes": len(episodes),
        "episode_ids": sorted(episodes),
        "languages": dict(sorted(languages.items())),
        "white_right_transitions": int(white_right),
        "geometry_shape": list(geometry.shape),
        "geometry_finite": bool(finite.all()),
        "finite_fraction": float(finite.mean()) if len(finite) else 0.0,
        "nonzero_delta_fraction": float(nonzero.mean()) if len(nonzero) else 0.0,
        "coverage": float(valid.mean()) if len(valid) else 0.0,
        "in_grid_fraction": float(in_grid.mean()) if len(in_grid) else 0.0,
        "out_of_grid_count": int((~in_grid).sum()),
        "geometry_min": float(np.nanmin(geometry)),
        "geometry_max": float(np.nanmax(geometry)),
        "delta_norm_min": float(np.nanmin(delta_norm)),
        "delta_norm_max": float(np.nanmax(delta_norm)),
        "router": cache["summary"].get("router"),
        "router_sha256": cache["summary"].get("router_sha256"),
        "checkpoint_fingerprint": cache["summary"].get("checkpoint_fingerprint"),
        "geometry_note_present": any(
            "geometry is computed from route weights" in str(note)
            for note in cache["summary"].get("notes", [])
        ),
    }


def summarize_inventory(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    splits = payload["splits"]
    result: dict[str, Any] = {}
    for name, rows in splits.items():
        result[name] = {
            "episodes": len(rows),
            "episode_ids": sorted({str(row["episode_id"]) for row in rows}),
            "init_state_shas": sorted({str(row["init_state_sha256"]) for row in rows}),
            "languages": dict(sorted(Counter(str(row.get("language")) for row in rows).items())),
        }
    return {"path": str(path), "counts": payload.get("counts"), "splits": result}


def p0_white_right_rows(path: Path) -> dict[str, Any]:
    rows = [row for row in read_jsonl(path) if is_white_right(row.get("language"))]
    return {
        "path": str(path),
        "rows": len(rows),
        "shared_init_ids": sorted({str(row["shared_init_id"]) for row in rows}),
        "init_state_shas": sorted({str(row["init_state_sha256"]) for row in rows}),
    }


def intersection(left: list[str] | set[str], right: list[str] | set[str]) -> list[str]:
    return sorted(set(left) & set(right))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--p0-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train = summarize_geometry(args.train_cache)
    validation = summarize_geometry(args.validation_cache)
    inventory = summarize_inventory(args.inventory)
    p0 = p0_white_right_rows(args.p0_manifest)
    inv_splits = inventory["splits"]
    split_names = sorted(inv_splits)
    split_overlap: dict[str, list[str]] = {}
    init_overlap: dict[str, list[str]] = {}
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            key = f"{left}__{right}"
            split_overlap[key] = intersection(
                inv_splits[left]["episode_ids"], inv_splits[right]["episode_ids"]
            )
            init_overlap[key] = intersection(
                inv_splits[left]["init_state_shas"], inv_splits[right]["init_state_shas"]
            )

    train_ids = set(inv_splits.get("train", {}).get("episode_ids", []))
    validation_ids = set(inv_splits.get("validation", {}).get("episode_ids", []))
    p0_init_overlap_train = intersection(p0["init_state_shas"], inv_splits["train"]["init_state_shas"])
    p0_init_overlap_validation = intersection(
        p0["init_state_shas"], inv_splits["validation"]["init_state_shas"]
    )
    p0_init_overlap_rollout = intersection(
        p0["init_state_shas"], inv_splits["rollout"]["init_state_shas"]
    )
    cache_train_ids = set(train["episode_ids"])
    cache_validation_ids = set(validation["episode_ids"])
    checks = {
        "train_cache_matches_inventory": cache_train_ids == train_ids,
        "validation_cache_matches_inventory": cache_validation_ids == validation_ids,
        "train_validation_episode_disjoint": not split_overlap.get("train__validation", []),
        "train_rollout_episode_disjoint": not split_overlap.get("train__rollout", []),
        "validation_rollout_episode_disjoint": not split_overlap.get("validation__rollout", []),
        "train_validation_init_disjoint": not init_overlap.get("train__validation", []),
        "train_rollout_init_disjoint": not init_overlap.get("train__rollout", []),
        "validation_rollout_init_disjoint": not init_overlap.get("validation__rollout", []),
        "p0_eval_disjoint_from_train": not p0_init_overlap_train,
        "p0_eval_disjoint_from_validation": not p0_init_overlap_validation,
        "p0_eval_matches_rollout_source": bool(p0_init_overlap_rollout),
        "white_right_train_zero": train["white_right_transitions"] == 0,
        "train_geometry_finite": train["geometry_finite"],
        "validation_geometry_finite": validation["geometry_finite"],
        "train_coverage_ge_090": train["coverage"] >= 0.90,
        "validation_coverage_ge_090": validation["coverage"] >= 0.90,
        "router_sha_recorded": bool(train["router_sha256"] and validation["router_sha256"]),
        "geometry_provenance_recorded": bool(
            train["geometry_note_present"] and validation["geometry_note_present"]
        ),
    }
    status = "PASS" if all(checks.values()) else "INVALID"
    payload = {
        "status": status,
        "run_contract_status": status,
        "evidence_type": "route_centroid",
        "arm_name": "R2-ROUTE",
        "point_in_mask": None,
        "reason_point_in_mask_not_applicable": (
            "P1-A selects frozen T1 route-centroid geometry; no J2 point head is used."
        ),
        "gates": checks,
        "train": {key: value for key, value in train.items() if key != "episode_ids"},
        "validation": {key: value for key, value in validation.items() if key != "episode_ids"},
        "inventory": inventory,
        "p0_white_right": p0,
        "split_episode_overlap": split_overlap,
        "split_init_state_overlap": init_overlap,
        "p0_white_right_init_overlap": {
            "train": p0_init_overlap_train,
            "validation": p0_init_overlap_validation,
            "rollout": p0_init_overlap_rollout,
        },
        "sha256": {
            "train_cache_summary": sha256_file(args.train_cache / "summary.json"),
            "validation_cache_summary": sha256_file(args.validation_cache / "summary.json"),
            "inventory": sha256_file(args.inventory),
            "p0_manifest": sha256_file(args.p0_manifest),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in ("status", "evidence_type", "arm_name", "gates")}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
