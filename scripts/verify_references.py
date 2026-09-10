#!/usr/bin/env python3
"""Verify the paper and repository archive without third-party dependencies."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "references" / "papers"
REPOS = ROOT / "references" / "repos"

REQUIRED_PAPERS = [
    "lavender_2502.06814.pdf",
    "spikingbrain_2509.05276.pdf",
    "qwen25_vl_2502.13923.pdf",
    "qwen2_vl_2409.12191.pdf",
    "qwen3_vl_2511.21631.pdf",
    "deepseek_vl2_2412.10302.pdf",
    "openvla_2406.09246.pdf",
    "pi0_2410.24164.pdf",
    "qwen_robotmanip_2606.17846.pdf",
    "lingbot_video_2607.07675.pdf",
    "lingbot_va2_2607.08639.pdf",
    "lingbot_vla2_2607.06403.pdf",
    "libero_2306.03310.pdf",
    "diffusion_policy_2303.04137.pdf",
    "gla_2312.06635.pdf",
    "attention_sinks_2309.17453.pdf",
]

REQUIRED_REPOS = [
    "lavender",
    "spikingbrain-7b",
    "openvla-oft",
    "qwen2.5-vl",
    "qwen3-vl",
    "deepseek-vl2",
    "openvla",
    "qwen-robotmanip",
    "openpi",
    "libero",
    "diffusion_policy",
    "lingbot-va",
    "lingbot-vla",
    "lingbot-vla-v2",
]

# Official local reports can be useful substitutes for an exact arXiv PDF,
# but they do not make the exact paper download complete.
PAPER_SUBSTITUTES = {
    "spikingbrain_2509.05276.pdf": "SpikingBrain_Report_Eng.pdf",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    available_papers = []
    missing_papers = []
    substituted_papers = []
    for name in REQUIRED_PAPERS:
        path = PAPERS / name
        if path.is_file() and path.stat().st_size > 100_000:
            available_papers.append(name)
        elif (PAPERS / PAPER_SUBSTITUTES.get(name, "")).is_file():
            substituted_papers.append((name, PAPER_SUBSTITUTES[name]))
        else:
            missing_papers.append(name)

    available_repos = []
    missing_repos = []
    for name in REQUIRED_REPOS:
        path = REPOS / name
        if (path / ".git").is_dir():
            available_repos.append(name)
        else:
            missing_repos.append(name)

    checksum_file = ROOT / "references" / "checksums.sha256"
    checksum_failures = []
    if checksum_file.is_file():
        for line in checksum_file.read_text().splitlines():
            if not line.strip():
                continue
            expected, relative = line.split(maxsplit=1)
            path = ROOT / "references" / relative
            if not path.is_file() or sha256(path) != expected:
                checksum_failures.append(relative)

    print(f"papers: {len(available_papers)}/{len(REQUIRED_PAPERS)} available")
    print(f"paper substitutes: {len(substituted_papers)}")
    print(f"repos:  {len(available_repos)}/{len(REQUIRED_REPOS)} available")
    print(f"checksum failures: {len(checksum_failures)}")
    if missing_papers:
        print("missing papers:")
        for name in missing_papers:
            print(f"  - {name}")
    if substituted_papers:
        print("paper substitutes (exact PDF still missing):")
        for requested, substitute in substituted_papers:
            print(f"  - {requested} <- {substitute}")
    if missing_repos:
        print("missing repos:")
        for name in missing_repos:
            print(f"  - {name}")
    if checksum_failures:
        print("checksum failures:")
        for name in checksum_failures:
            print(f"  - {name}")

    return 0 if not missing_papers and not missing_repos and not checksum_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
