"""CPU-only comparison of shared-pair pilot maps against row-wise references."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()

    group_file = args.pilot_dir / "pilot_shared_groups.jsonl"
    groups = [
        json.loads(line)
        for line in group_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not groups:
        raise SystemExit("empty pilot group manifest")

    rows: list[dict[str, Any]] = []
    all_passed = True
    for group in groups:
        member_results = []
        for member in group["members"]:
            sample_id = member["sample_id"]
            pilot_map = args.pilot_dir / f"{sample_id}.npy"
            pilot_meta = args.pilot_dir / f"{sample_id}.json"
            reference_map = args.reference_cache / f"{sample_id}.npy"
            reference_meta = args.reference_cache / f"{sample_id}.json"
            result: dict[str, Any] = {
                "sample_id": sample_id,
                "role": member.get("role"),
                "pilot_map_exists": pilot_map.is_file(),
                "pilot_metadata_exists": pilot_meta.is_file(),
                "reference_map_exists": reference_map.is_file(),
                "reference_metadata_exists": reference_meta.is_file(),
            }
            if not all(
                result[key]
                for key in (
                    "pilot_map_exists",
                    "pilot_metadata_exists",
                    "reference_map_exists",
                    "reference_metadata_exists",
                )
            ):
                result["passed"] = False
                result["reason"] = "missing map or metadata"
                all_passed = False
                member_results.append(result)
                continue

            pmap = np.load(pilot_map)
            rmap = np.load(reference_map)
            pmeta = json.loads(pilot_meta.read_text(encoding="utf-8"))
            rmeta = json.loads(reference_meta.read_text(encoding="utf-8"))
            finite = bool(np.isfinite(pmap).all() and np.isfinite(rmap).all())
            shape_equal = pmap.shape == rmap.shape == (16, 16)
            max_abs = float(np.max(np.abs(pmap - rmap))) if shape_equal else None
            mean_abs = float(np.mean(np.abs(pmap - rmap))) if shape_equal else None
            close = bool(
                shape_equal
                and finite
                and np.allclose(pmap, rmap, atol=args.atol, rtol=args.rtol)
            )
            metadata_ok = bool(
                pmeta.get("method") == "ddim_nulltext_fp32_reconstruction_v3"
                and pmeta.get("capture_stage") == "final_reconstruction_only"
                and pmeta.get("pilot_only") is True
                and pmeta.get("attention_tensors") == 100
                and pmeta.get("sample_id") == sample_id
                and pmeta.get("role") == member.get("role")
                and pmeta.get("phrase") == member.get("phrase")
                and rmeta.get("sample_id") == sample_id
                and rmeta.get("phrase") == member.get("phrase")
            )
            result.update(
                {
                    "pilot_shape": list(pmap.shape),
                    "reference_shape": list(rmap.shape),
                    "finite": finite,
                    "shape_equal": shape_equal,
                    "max_abs_error": max_abs,
                    "mean_abs_error": mean_abs,
                    "allclose": close,
                    "metadata_ok": metadata_ok,
                    "pilot_map_sha256": sha256(pilot_map),
                    "reference_map_sha256": sha256(reference_map),
                    "passed": bool(close and metadata_ok),
                }
            )
            if not result["passed"]:
                all_passed = False
            member_results.append(result)
        rows.append(
            {
                "group_id": group["group_id"],
                "members": member_results,
                "passed": all(item["passed"] for item in member_results),
            }
        )

    report = {
        "pilot_dir": str(args.pilot_dir),
        "reference_cache": str(args.reference_cache),
        "groups": rows,
        "group_count": len(rows),
        "passed": all_passed,
        "atol": args.atol,
        "rtol": args.rtol,
        "comparison": "shared capture map versus existing row-wise map",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
