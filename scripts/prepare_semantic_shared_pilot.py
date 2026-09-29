"""Prepare an isolated semantic source/target sharing pilot manifest.

This utility is CPU-only. It never edits a formal cache and never emits maps.
It groups LIBERO semantic rows by the full teacher computation key while
retaining phrase, role, occurrence, token span and sample_id as per-member
metadata. The resulting pilot manifest is intended for a separate GPU runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _value(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def _canonical(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value


def shared_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Return the key for one shared full-caption teacher computation."""
    return (
        _canonical(_value(row, "image_sha256", "image_sha", "image_path", "image")),
        _canonical(_value(row, "caption", "instruction", "text")),
        _canonical(_value(row, "camera", "camera_name")),
        _canonical(_value(row, "episode_id", "episode")),
        _canonical(_value(row, "timestep", "time_step", "frame_id")),
        _canonical(_value(row, "teacher_model", "model")),
        _canonical(_value(row, "teacher_revision", "revision")),
        _canonical(_value(row, "seed")),
        _canonical(_value(row, "inversion_steps", "steps")),
        _canonical(_value(row, "inner_steps")),
        _canonical(_value(row, "guidance_scale", "cfg", "cfg_scale")),
        _canonical(_value(row, "resolution", "image_resolution")),
        _canonical(_value(row, "numeric_backend", "attention_backend", "dtype")),
        _canonical(_value(row, "extractor_sha256", "extractor_sha")),
    )


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=2)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit("empty manifest")

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        required = ("sample_id", "image_path", "caption", "phrase", "role")
        missing = [key for key in required if not row.get(key)]
        if missing:
            raise SystemExit(f"manifest row {row.get('sample_id')} missing {missing}")
        groups[shared_key(row)].append(row)

    candidates: list[tuple[tuple[Any, ...], list[dict[str, Any]]]] = []
    for key, members in groups.items():
        roles = {str(member.get("role")) for member in members}
        if len(members) == 2 and roles == {"source", "target"}:
            candidates.append((key, members))
    candidates.sort(key=lambda item: str(item[1][0].get("sample_id")))
    if len(candidates) < args.pairs:
        raise SystemExit(f"only {len(candidates)} valid source/target pairs, need {args.pairs}")

    selected = candidates[: args.pairs]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    extractor_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    group_lines: list[str] = []
    member_lines: list[str] = []
    selected_report: list[dict[str, Any]] = []
    for index, (key, members) in enumerate(selected):
        key_text = json.dumps(key, ensure_ascii=False, sort_keys=False, separators=(",", ":"))
        group_id = hashlib.sha256(key_text.encode("utf-8")).hexdigest()
        ordered = sorted(members, key=lambda member: str(member["role"]))
        group = {
            "group_id": group_id,
            "group_index": index,
            "shared_key": list(key),
            "manifest_sha256": manifest_sha,
            "extractor_sha256": extractor_sha,
            "members": ordered,
        }
        group_lines.append(json.dumps(group, ensure_ascii=False, sort_keys=True))
        for member in ordered:
            member_copy = dict(member)
            member_copy["shared_group_id"] = group_id
            member_copy["pilot_group_index"] = index
            member_lines.append(json.dumps(member_copy, ensure_ascii=False, sort_keys=True))
        selected_report.append(
            {
                "group_id": group_id,
                "sample_ids": [member["sample_id"] for member in ordered],
                "roles": [member["role"] for member in ordered],
                "phrases": [member["phrase"] for member in ordered],
            }
        )

    report = {
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_sha,
        "extractor_sha256": extractor_sha,
        "row_count": len(rows),
        "shared_group_count": len(groups),
        "valid_source_target_pair_count": len(candidates),
        "selected_pair_count": len(selected),
        "key_excludes": ["phrase", "phrase_occurrence", "role", "sample_id", "token_span"],
        "member_metadata_retained": [
            "sample_id",
            "role",
            "phrase",
            "phrase_occurrence",
            "token_span",
            "image_path",
            "caption",
            "instruction",
            "episode_id",
            "timestep",
            "camera",
        ],
        "selected_pairs": selected_report,
    }
    atomic_write(output / "pilot_shared_groups.jsonl", "\n".join(group_lines) + "\n")
    atomic_write(output / "pilot_member_manifest.jsonl", "\n".join(member_lines) + "\n")
    atomic_write(output / "pilot_manifest_report.json", json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
