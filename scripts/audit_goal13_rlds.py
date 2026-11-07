#!/usr/bin/env python3
"""Read-only Goal13 RLDS audit without TensorFlow.

The LIBERO RLDS files are TFRecord streams.  Stage 0 only needs episode-level
language and split membership, so this script reads the TFRecord framing and
matches the known instruction bytes directly.  It deliberately does not decode
images, actions, or model inputs, and it never writes to the source dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "goal13-rlds-audit-v1"
PROMPTS = {
    "red_left": b"put the red mug on the left plate",
    "red_right": b"put the red mug on the right plate",
    "white_left": b"put the white mug on the left plate",
    "white_right": b"put the white mug on the right plate",
    "yellow_and_white_right": b"put the yellow and white mug on the right plate",
    "white_bowl_right": b"put the white bowl to the right of the plate",
}
PROMPT_TEXT = {name: token.decode("utf-8") for name, token in PROMPTS.items()}
SPLITS = ("train", "validation", "rollout")
DEFAULT_EXPECTED_SOURCE_COUNTS = {
    "red_left": 37,
    "red_right": 44,
    "white_left": 45,
    "white_right": 0,
    "yellow_and_white_right": 48,
    "white_bowl_right": 44,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def iter_tfrecord_payloads(path: Path) -> Iterable[bytes]:
    """Yield raw TFRecord payloads, validating length framing only.

    TensorFlow's masked CRC32C is intentionally not checked here because the
    current remote CPU environment has neither TensorFlow nor a CRC32C package.
    A truncated or malformed record still fails closed on the length check.
    """

    with path.open("rb") as handle:
        while True:
            header = handle.read(8)
            if not header:
                return
            if len(header) != 8:
                raise ValueError(f"truncated TFRecord length header: {path}")
            (length,) = struct.unpack("<Q", header)
            length_crc = handle.read(4)
            if len(length_crc) != 4:
                raise ValueError(f"missing TFRecord length CRC: {path}")
            payload = handle.read(length)
            if len(payload) != length:
                raise ValueError(f"truncated TFRecord payload: {path}")
            payload_crc = handle.read(4)
            if len(payload_crc) != 4:
                raise ValueError(f"missing TFRecord payload CRC: {path}")
            yield payload


def detect_prompt(payload: bytes) -> str | None:
    matches = [name for name, token in PROMPTS.items() if token in payload]
    if len(matches) > 1:
        raise ValueError(f"record contains multiple Goal13 prompts: {matches}")
    return matches[0] if matches else None


def load_split_manifest(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    payload = json.loads(path.read_text())
    rows_by_id: dict[str, dict[str, Any]] = {}
    ids_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    for split, rows in payload.get("splits", {}).items():
        if split not in ids_by_split:
            raise ValueError(f"unknown split in manifest: {split}")
        for row in rows:
            episode_id = str(row["episode_id"])
            if episode_id in rows_by_id:
                raise ValueError(f"duplicate episode in split manifest: {episode_id}")
            enriched = dict(row)
            enriched["split"] = split
            rows_by_id[episode_id] = enriched
            ids_by_split[split].add(episode_id)
    return rows_by_id, ids_by_split


def _write_progress(
    path: Path | None,
    *,
    status: str,
    shards_done: int,
    shards_total: int,
    records_scanned: int,
    target_records: int,
) -> None:
    if path is None:
        return
    atomic_json_write(
        path,
        {
            "schema": "goal13-audit-progress-v1",
            "status": status,
            "shards_done": shards_done,
            "shards_total": shards_total,
            "records_scanned": records_scanned,
            "target_records": target_records,
            "updated_at": utc_now(),
        },
    )


def audit_shards(
    shards: list[Path],
    manifest_by_id: dict[str, dict[str, Any]],
    *,
    progress_path: Path | None = None,
) -> dict[str, Any]:
    source_counts: Counter[str] = Counter({name: 0 for name in PROMPTS})
    split_counts: dict[str, Counter[str]] = {
        split: Counter({name: 0 for name in PROMPTS}) for split in (*SPLITS, "unassigned")
    }
    episodes: list[dict[str, Any]] = []
    duplicate_ids: list[str] = []
    seen_ids: set[str] = set()
    records_scanned = 0

    for shard_index, shard in enumerate(shards):
        for record_index, payload in enumerate(iter_tfrecord_payloads(shard)):
            records_scanned += 1
            episode_id = sha256_bytes(payload)
            if episode_id in seen_ids:
                duplicate_ids.append(episode_id)
            seen_ids.add(episode_id)

            prompt = detect_prompt(payload)
            if prompt is None:
                continue
            manifest_row = manifest_by_id.get(episode_id)
            split = str(manifest_row["split"]) if manifest_row else "unassigned"
            source_counts[prompt] += 1
            split_counts[split][prompt] += 1
            episodes.append(
                {
                    "episode_id": episode_id,
                    "prompt": prompt,
                    "language": str(manifest_row["language"]) if manifest_row else PROMPT_TEXT[prompt],
                    "split": split,
                    "source_shard": str(shard),
                    "source_shard_index": shard_index,
                    "source_record_index": record_index,
                }
            )
        _write_progress(
            progress_path,
            status="running",
            shards_done=shard_index + 1,
            shards_total=len(shards),
            records_scanned=records_scanned,
            target_records=len(episodes),
        )

    return {
        "source_counts": dict(source_counts),
        "split_counts": {split: dict(counts) for split, counts in split_counts.items()},
        "episodes": episodes,
        "duplicate_episode_ids": duplicate_ids,
        "records_scanned": records_scanned,
    }


def _pairwise_overlap(ids_by_split: dict[str, set[str]]) -> dict[str, int]:
    names = list(ids_by_split)
    return {
        f"{left}__{right}": len(ids_by_split[left] & ids_by_split[right])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    }


def build_report(
    *,
    data_path: Path,
    shard_count: int,
    expected_shards: int,
    audit: dict[str, Any],
    manifest_by_id: dict[str, dict[str, Any]],
    ids_by_split: dict[str, set[str]],
    expected_source_counts: dict[str, int],
) -> dict[str, Any]:
    source_counts = {name: int(audit["source_counts"].get(name, 0)) for name in PROMPTS}
    train_language_counts = {
        name: int(audit["split_counts"]["train"].get(name, 0)) for name in ("red_left", "red_right", "white_left")
    }
    found_manifest_ids = {row["episode_id"] for row in audit["episodes"] if row["episode_id"] in manifest_by_id}
    missing_manifest_ids = sorted(set(manifest_by_id) - found_manifest_ids)
    language_mismatches = sorted(
        row["episode_id"]
        for row in audit["episodes"]
        if row["episode_id"] in manifest_by_id
        and row["language"] != manifest_by_id[row["episode_id"]].get("language")
    )
    overlap = _pairwise_overlap(ids_by_split)
    overlap_total = sum(overlap.values())
    manifest_split_counts = {split: len(ids_by_split[split]) for split in SPLITS}
    manifest_language_counts = Counter(str(row.get("language", "")) for row in manifest_by_id.values())
    expected_language_counts = {
        "red_left": expected_source_counts["red_left"],
        "red_right": expected_source_counts["red_right"],
        "white_left": expected_source_counts["white_left"],
    }
    manifest_expected_language_counts = {
        PROMPT_TEXT[name]: count for name, count in expected_language_counts.items()
    }

    checks = {
        "expected_shards": {
            "passed": shard_count == expected_shards,
            "expected": expected_shards,
            "actual": shard_count,
        },
        "duplicate_episode_ids": {
            "passed": not audit["duplicate_episode_ids"],
            "expected": 0,
            "actual": len(audit["duplicate_episode_ids"]),
        },
        "manifest_ids_covered": {
            "passed": not missing_manifest_ids,
            "expected": 0,
            "actual": len(missing_manifest_ids),
            "missing": missing_manifest_ids[:20],
        },
        "manifest_language_match": {
            "passed": not language_mismatches,
            "expected": 0,
            "actual": len(language_mismatches),
            "mismatches": language_mismatches[:20],
        },
        "manifest_split_counts": {
            "passed": manifest_split_counts == {"train": 96, "validation": 15, "rollout": 15},
            "expected": {"train": 96, "validation": 15, "rollout": 15},
            "actual": manifest_split_counts,
        },
        "manifest_language_counts": {
            "passed": dict(manifest_language_counts) == manifest_expected_language_counts,
            "expected": manifest_expected_language_counts,
            "actual": dict(manifest_language_counts),
        },
        "split_overlap_zero": {
            "passed": overlap_total == 0,
            "expected": 0,
            "actual": overlap_total,
            "pairwise": overlap,
        },
        "white_right_train_zero": {
            "passed": int(audit["split_counts"]["train"].get("white_right", 0)) == 0,
            "expected": 0,
            "actual": int(audit["split_counts"]["train"].get("white_right", 0)),
        },
        "source_counts": {
            "passed": source_counts == expected_source_counts,
            "expected": expected_source_counts,
            "actual": source_counts,
        },
    }
    failures = [name for name, check in checks.items() if not check["passed"]]
    return {
        "schema": SCHEMA,
        "created_at": utc_now(),
        "audit_status": "PASS" if not failures else "INVALID",
        "source": {
            "data_path": str(data_path),
            "shard_count": shard_count,
            "expected_shards": expected_shards,
            "records_scanned": int(audit["records_scanned"]),
            "target_records": len(audit["episodes"]),
            "crc_check": "not_checked",
        },
        "source_language_counts": source_counts,
        "train_language_counts": train_language_counts,
        "split_language_counts": audit["split_counts"],
        "manifest_split_counts": manifest_split_counts,
        "manifest_language_counts": dict(manifest_language_counts),
        "split_overlap": overlap,
        "split_overlap_total": overlap_total,
        "white_right_train_count": int(audit["split_counts"]["train"].get("white_right", 0)),
        "carrier_episode_count": source_counts["yellow_and_white_right"],
        "white_bowl_right_episode_count": source_counts["white_bowl_right"],
        "checks": checks,
        "failures": failures,
        "episodes": sorted(
            audit["episodes"],
            key=lambda row: (row["split"], row["prompt"], row["source_shard_index"], row["source_record_index"]),
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path, default=None)
    parser.add_argument("--shard-glob", default="*.tfrecord-*")
    parser.add_argument("--expected-shards", type=int, default=256)
    parser.add_argument("--expected-red-left", type=int, default=DEFAULT_EXPECTED_SOURCE_COUNTS["red_left"])
    parser.add_argument("--expected-red-right", type=int, default=DEFAULT_EXPECTED_SOURCE_COUNTS["red_right"])
    parser.add_argument("--expected-white-left", type=int, default=DEFAULT_EXPECTED_SOURCE_COUNTS["white_left"])
    parser.add_argument("--expected-white-right", type=int, default=DEFAULT_EXPECTED_SOURCE_COUNTS["white_right"])
    parser.add_argument(
        "--expected-carrier",
        type=int,
        default=DEFAULT_EXPECTED_SOURCE_COUNTS["yellow_and_white_right"],
    )
    parser.add_argument(
        "--expected-white-bowl",
        type=int,
        default=DEFAULT_EXPECTED_SOURCE_COUNTS["white_bowl_right"],
    )
    args = parser.parse_args()

    shards = sorted(args.data.glob(args.shard_glob))
    if not shards:
        print(f"ERROR: no shards found under {args.data} matching {args.shard_glob}", file=sys.stderr)
        return 2
    manifest_by_id, ids_by_split = load_split_manifest(args.split_manifest)
    expected_counts = {
        "red_left": args.expected_red_left,
        "red_right": args.expected_red_right,
        "white_left": args.expected_white_left,
        "white_right": args.expected_white_right,
        "yellow_and_white_right": args.expected_carrier,
        "white_bowl_right": args.expected_white_bowl,
    }

    try:
        audit = audit_shards(shards, manifest_by_id, progress_path=args.progress)
        report = build_report(
            data_path=args.data,
            shard_count=len(shards),
            expected_shards=args.expected_shards,
            audit=audit,
            manifest_by_id=manifest_by_id,
            ids_by_split=ids_by_split,
            expected_source_counts=expected_counts,
        )
        atomic_json_write(args.output, report)
        _write_progress(
            args.progress,
            status="complete" if report["audit_status"] == "PASS" else "invalid",
            shards_done=len(shards),
            shards_total=len(shards),
            records_scanned=int(audit["records_scanned"]),
            target_records=len(audit["episodes"]),
        )
    except Exception as error:
        _write_progress(
            args.progress,
            status="failed",
            shards_done=0,
            shards_total=len(shards),
            records_scanned=0,
            target_records=0,
        )
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "audit_status": report["audit_status"],
                "source_language_counts": report["source_language_counts"],
                "train_language_counts": report["train_language_counts"],
                "split_overlap_total": report["split_overlap_total"],
                "failures": report["failures"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if report["audit_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
