"""Merge disjoint teacher-cache shards with manifest-level coverage checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--shard", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = [json.loads(x) for x in args.manifest.read_text().splitlines() if x.strip()]
    expected = [str(r["sample_id"]) for r in rows]
    if len(expected) != len(set(expected)):
        raise SystemExit("manifest has duplicate sample_id")
    args.output.mkdir(parents=True, exist_ok=True)
    entries = []
    missing, duplicate = [], []
    for sample_id in expected:
        key = sample_id.replace(":", "_")
        hits = []
        for shard in args.shard:
            npy, meta = shard / f"{key}.npy", shard / f"{key}.json"
            if npy.exists() or meta.exists():
                hits.append((shard, npy, meta))
        if len(hits) != 1:
            (duplicate if len(hits) > 1 else missing).append(sample_id)
            continue
        shard, npy, meta = hits[0]
        if not (npy.exists() and meta.exists()):
            missing.append(sample_id)
            continue
        dst_npy, dst_meta = args.output / npy.name, args.output / meta.name
        if not dst_npy.exists(): shutil.copy2(npy, dst_npy)
        if not dst_meta.exists(): shutil.copy2(meta, dst_meta)
        entries.append({"sample_id": sample_id, "shard": str(shard), "map": dst_npy.name,
                        "map_sha256": sha256(dst_npy), "metadata": dst_meta.name,
                        "metadata_sha256": sha256(dst_meta)})
    if missing or duplicate or len(entries) != len(expected):
        (args.output / "merge_failures.json").write_text(json.dumps({"missing": missing, "duplicate": duplicate}, indent=2) + "\n")
        raise SystemExit(f"cache coverage failed: entries={len(entries)} expected={len(expected)} missing={len(missing)} duplicate={len(duplicate)}")
    manifest_path = args.output / "cache_manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries))
    (args.output / "cache_manifest.sha256").write_text(sha256(manifest_path) + "\n")
    print(json.dumps({"entries": len(entries), "output": str(args.output), "manifest_sha256": sha256(manifest_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
