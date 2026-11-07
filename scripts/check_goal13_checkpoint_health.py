#!/usr/bin/env python3
"""Emit SHA256 and finite-value evidence for a Goal13 safetensors checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "goal13-checkpoint-health-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    import numpy as np
    from safetensors import safe_open

    started = time.monotonic()
    tensor_count = 0
    float_tensor_count = 0
    numel = 0
    nonfinite_tensors: list[str] = []
    with safe_open(str(path), framework="np") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            tensor_count += 1
            numel += int(tensor.size)
            if np.issubdtype(tensor.dtype, np.floating):
                float_tensor_count += 1
                if not bool(np.isfinite(tensor).all()):
                    nonfinite_tensors.append(key)
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "tensor_count": tensor_count,
        "float_tensor_count": float_tensor_count,
        "numel": numel,
        "nonfinite_tensors": nonfinite_tensors,
        "all_finite": float_tensor_count > 0 and not nonfinite_tensors,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {args.checkpoint}")
    payload = inspect_checkpoint(args.checkpoint)
    atomic_json_write(args.output, payload)
    print(
        json.dumps(
            {
                "all_finite": payload["all_finite"],
                "tensor_count": payload["tensor_count"],
                "nonfinite_tensors": payload["nonfinite_tensors"],
                "sha256": payload["sha256"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if payload["all_finite"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
