"""Validate a native-model P1 report before it is used in a training decision."""
import argparse
from pathlib import Path

from vla_attention.audit_io import read_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--expected-samples", type=int, default=20)
    parser.add_argument("--minimum-repeatability", type=float, default=0.90)
    args = parser.parse_args()

    capability, rows = read_report(args.report)
    ids = {row.sample_id for row in rows}
    if len(ids) < args.expected_samples:
        raise SystemExit(f"FAIL: only {len(ids)} unique samples; expected {args.expected_samples}")
    nonfinite = [row.sample_id for row in rows if not row.gradient_finite]
    if nonfinite:
        raise SystemExit(f"FAIL: non-finite gradients for {len(nonfinite)} samples")
    unstable = [row.sample_id for row in rows if row.repeatability is not None and row.repeatability < args.minimum_repeatability]
    if unstable:
        raise SystemExit(f"FAIL: repeatability below {args.minimum_repeatability} for {len(unstable)} samples")
    print(f"PASS {capability.model_id}@{capability.model_revision}: {len(rows)} measurements / {len(ids)} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
