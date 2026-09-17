"""Validate tracked JSON experiment manifests without ML dependencies."""
import json
import sys
from pathlib import Path

REQUIRED = {"experiment_id", "stage", "hypothesis", "model", "teacher", "data", "targets", "budget", "stopping_rules", "metrics", "seed"}

def validate(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    errors = []
    missing = REQUIRED - set(data)
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
    if not isinstance(data.get("targets"), list) or not data.get("targets"):
        errors.append("targets must be a non-empty list")
    if not isinstance(data.get("stopping_rules"), list) or not data.get("stopping_rules"):
        errors.append("stopping_rules must be a non-empty list")
    if not isinstance(data.get("model"), dict) or not data.get("model", {}).get("id"):
        errors.append("model.id is required")
    return errors

if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]] or sorted(Path("configs/experiments").glob("*.json"))
    failed = False
    for path in paths:
        errors = validate(path)
        if errors:
            failed = True
            print(f"FAIL {path}: {'; '.join(errors)}")
        else:
            print(f"PASS {path}")
    raise SystemExit(1 if failed else 0)
