"""Minimal deterministic JSONL loaders used by VLM experiment runners."""
from __future__ import annotations

import json
from pathlib import Path


def jsonl(path: Path):
    for line in path.read_text().splitlines():
        if line.strip():
            yield json.loads(line)
