"""Render report.md without nested list blocks for Feishu's Markdown converter."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = (ROOT / 'report.md').read_text()
lines = []
for line in source.splitlines():
    # The current converter creates descendants for nested bullets, while its
    # verifier reads only top-level text. Use explicit paragraphs so the full
    # audit remains visible and verifiable online.
    if line.startswith('  - '):
        lines.append(line[4:])
    else:
        lines.append(line)
output = '\n'.join(lines) + '\n'
destination = ROOT / 'report_feishu.md'
destination.write_text(output)
print(destination)
