#!/usr/bin/env python3
"""Extract the Goal13 carrier episodes into an ordered TFRecord split.

The carrier records are scattered across the original LIBERO-90 shards.  The
LeRobot converter expects a TFRecord whose record order exactly matches its
episode manifest, so this tool copies only the audited carrier records and
rewrites ``output_record_index`` in a deterministic order.  It fails closed on
count, language, hash, duplicate, or split-overlap violations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.audit_goal13_rlds import (
        PROMPTS,
        PROMPT_TEXT,
        SPLITS,
        iter_tfrecord_payloads,
        sha256_bytes,
    )
except ImportError:  # Support running from the flat /tmp/goal13_tools upload.
    from audit_goal13_rlds import (  # type: ignore[no-redef]
        PROMPTS,
        PROMPT_TEXT,
        SPLITS,
        iter_tfrecord_payloads,
        sha256_bytes,
    )


SCHEMA = "goal13-carrier-extract-v1"
CARRIER_PROMPT = "yellow_and_white_right"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_tfrecord_record(handle: Any, payload: bytes) -> None:
    handle.write(struct.pack("<Q", len(payload)))
    handle.write(b"\x00\x00\x00\x00")
    handle.write(payload)
    handle.write(b"\x00\x00\x00\x00")


def load_carrier_rows(audit_path: Path, *, expected_count: int) -> list[dict[str, Any]]:
    payload = json.loads(audit_path.read_text())
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"audit has no episodes list: {audit_path}")
    rows = [dict(row) for row in episodes if row.get("prompt") == CARRIER_PROMPT]
    if len(rows) != expected_count:
        raise ValueError(f"carrier count mismatch: audit={len(rows)} expected={expected_count}")

    ids = [str(row.get("episode_id", "")) for row in rows]
    if any(not episode_id for episode_id in ids):
        raise ValueError("carrier audit row is missing episode_id")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate carrier episode_id in audit")

    expected_language = PROMPT_TEXT[CARRIER_PROMPT]
    for row in rows:
        if row.get("language") != expected_language:
            raise ValueError(
                f"carrier language mismatch for {row.get('episode_id')}: {row.get('language')!r}"
            )
        if row.get("split") not in (None, "unassigned"):
            raise ValueError(
                f"carrier episode already belongs to split {row.get('split')!r}: {row.get('episode_id')}"
            )
        if not row.get("source_shard"):
            raise ValueError(f"carrier episode missing source_shard: {row.get('episode_id')}")
        if row.get("source_record_index") is None:
            raise ValueError(f"carrier episode missing source_record_index: {row.get('episode_id')}")

    rows.sort(
        key=lambda row: (
            str(row["source_shard"]),
            int(row["source_record_index"]),
            str(row["episode_id"]),
        )
    )
    return rows


def load_split_ids(split_manifest_path: Path | None) -> dict[str, set[str]]:
    if split_manifest_path is None:
        return {split: set() for split in SPLITS}
    payload = json.loads(split_manifest_path.read_text())
    output = {split: set() for split in SPLITS}
    for split, rows in payload.get("splits", {}).items():
        if split not in output:
            raise ValueError(f"unknown split in manifest: {split}")
        for row in rows:
            output[split].add(str(row["episode_id"]))
    return output


def extract_carrier(
    *,
    audit_path: Path,
    output_tfrecord: Path,
    output_manifest: Path,
    output_report: Path,
    source_root: Path | None = None,
    split_manifest_path: Path | None = None,
    expected_count: int = 48,
    overwrite: bool = False,
) -> dict[str, Any]:
    audit_sha = sha256_file(audit_path)
    rows = load_carrier_rows(audit_path, expected_count=expected_count)
    split_ids = load_split_ids(split_manifest_path)
    carrier_ids = {str(row["episode_id"]) for row in rows}
    overlaps = {split: len(carrier_ids & ids) for split, ids in split_ids.items()}
    overlap_total = sum(overlaps.values())
    if overlap_total:
        raise ValueError(f"carrier episodes overlap C1 splits: {overlaps}")

    rows_by_shard: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    root = source_root.resolve() if source_root is not None else None
    for row in rows:
        shard = Path(str(row["source_shard"]))
        if root is not None:
            resolved = shard.resolve()
            try:
                resolved.relative_to(root)
            except ValueError as error:
                raise ValueError(f"carrier shard escapes source root: {shard}") from error
            shard = resolved
        if not shard.is_file():
            raise FileNotFoundError(f"carrier source shard is missing: {shard}")
        rows_by_shard[shard].append(row)

    output_tfrecord.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        for path in (output_tfrecord, output_manifest, output_report):
            if path.exists():
                raise FileExistsError(path)

    selected_payloads: list[bytes] = []
    output_rows: list[dict[str, Any]] = []
    for shard in sorted(rows_by_shard):
        payloads = list(iter_tfrecord_payloads(shard))
        for row in sorted(rows_by_shard[shard], key=lambda item: int(item["source_record_index"])):
            source_index = int(row["source_record_index"])
            if source_index < 0 or source_index >= len(payloads):
                raise ValueError(
                    f"source_record_index out of range for {shard}: "
                    f"{source_index} not in [0, {len(payloads)})"
                )
            payload = payloads[source_index]
            actual_sha = sha256_bytes(payload)
            if actual_sha != str(row["episode_id"]):
                raise ValueError(
                    f"carrier episode hash mismatch at {shard}:{source_index}: "
                    f"payload={actual_sha} audit={row['episode_id']}"
                )
            if PROMPTS[CARRIER_PROMPT] not in payload:
                raise ValueError(
                    f"carrier payload does not contain its language at {shard}:{source_index}"
                )
            selected_payloads.append(payload)
            output_rows.append(
                {
                    "output_record_index": len(output_rows),
                    "episode_id": str(row["episode_id"]),
                    "language": PROMPT_TEXT[CARRIER_PROMPT],
                    "prompt": CARRIER_PROMPT,
                    "source_shard": str(shard),
                    "source_shard_index": row.get("source_shard_index"),
                    "source_record_index": source_index,
                }
            )

    if len(selected_payloads) != expected_count:
        raise ValueError(
            f"selected carrier count mismatch: {len(selected_payloads)} != {expected_count}"
        )

    temporary = output_tfrecord.with_suffix(output_tfrecord.suffix + ".tmp")
    with temporary.open("wb") as handle:
        for payload in selected_payloads:
            _write_tfrecord_record(handle, payload)
    temporary.replace(output_tfrecord)
    _write_jsonl(output_manifest, output_rows)

    carrier_ids_sha = hashlib.sha256(
        "\n".join(sorted(carrier_ids)).encode("utf-8")
    ).hexdigest()
    report = {
        "schema": SCHEMA,
        "created_at": utc_now(),
        "status": "PASS",
        "prompt": CARRIER_PROMPT,
        "language": PROMPT_TEXT[CARRIER_PROMPT],
        "expected_count": expected_count,
        "carrier_count": len(output_rows),
        "carrier_episode_ids_sha256": carrier_ids_sha,
        "split_overlaps": overlaps,
        "split_overlap_total": overlap_total,
        "source_audit": str(audit_path),
        "source_audit_sha256": audit_sha,
        "split_manifest": str(split_manifest_path) if split_manifest_path else None,
        "output_tfrecord": str(output_tfrecord),
        "output_tfrecord_sha256": sha256_file(output_tfrecord),
        "output_manifest": str(output_manifest),
        "output_manifest_sha256": sha256_file(output_manifest),
        "rows": output_rows,
    }
    atomic_json_write(output_report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-tfrecord", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--expected-count", type=int, default=48)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = extract_carrier(
        audit_path=args.audit,
        output_tfrecord=args.output_tfrecord,
        output_manifest=args.output_manifest,
        output_report=args.output_report,
        source_root=args.source_root,
        split_manifest_path=args.split_manifest,
        expected_count=args.expected_count,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "carrier_count": report["carrier_count"],
                "split_overlaps": report["split_overlaps"],
                "output_tfrecord": report["output_tfrecord"],
                "output_manifest": report["output_manifest"],
                "output_report": str(args.output_report),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
