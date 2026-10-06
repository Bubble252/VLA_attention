#!/usr/bin/env python3
"""Validate the model-independent invariants of the P6 VLA config.

This check intentionally does not import torch or any external model package.
It is a launch guard for local and remote runners; model and LIBERO-specific
fields remain unresolved until P1 and manifest generation have completed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PLACEHOLDER_RE = re.compile(r"^(TO_FREEZE|TO_CREATE)(?:$|[_:])")


def _find_placeholders(value: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(_find_placeholders(item, f"{path}.{key}" if path else key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_find_placeholders(item, f"{path}[{index}]"))
    elif isinstance(value, str) and PLACEHOLDER_RE.search(value):
        found.append(f"{path}={value}")
    return found


def _expect(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_config(config: dict[str, Any], *, allow_template: bool = False) -> list[str]:
    errors: list[str] = []
    _expect(str(config.get("experiment_id", "")).startswith("P6-VLA-B0-B4"), "experiment_id must start with P6-VLA-B0-B4", errors)

    status = config.get("status")
    placeholders = _find_placeholders(config)
    if placeholders and not allow_template:
        errors.append(
            "unresolved placeholders remain; run P1/manifest freeze before launch: "
            + ", ".join(placeholders[:8])
            + (" ..." if len(placeholders) > 8 else "")
        )
    if status == "template_not_runnable_until_p1_and_manifest" and not allow_template:
        errors.append("config status explicitly forbids training launch before P1 and manifest freeze")

    action = config.get("action", {})
    _expect(action.get("reported_schema") == "delta_eef_7d", "reported action schema must be delta_eef_7d", errors)
    _expect(action.get("dimensions") == 7, "reported action dimensions must be 7", errors)
    _expect(
        action.get("order") == ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "dgrip"],
        "reported action order must be [dx,dy,dz,droll,dpitch,dyaw,dgrip]",
        errors,
    )

    training = config.get("training", {})
    _expect(training.get("seeds") == [17, 29, 41], "formal seeds must be [17, 29, 41]", errors)
    _expect(training.get("early_stopping") is False, "early_stopping must be false", errors)
    _expect(training.get("drop_last", False) is False, "drop_last must be false when specified", errors)

    expected = {
        "B0": (False, False, False),
        "B1": (True, False, False),
        "B2": (False, True, False),
        "B3": (True, True, False),
        "B4": (True, True, True),
    }
    matrix = config.get("matrix", {})
    _expect(set(matrix) == set(expected), "matrix must contain exactly B0, B1, B2, B3, B4", errors)
    for group, (retention, semantic, containment) in expected.items():
        item = matrix.get(group, {})
        _expect(item.get("action_loss") is True, f"{group}.action_loss must be true", errors)
        _expect(item.get("retention") is retention, f"{group}.retention invariant violated", errors)
        _expect(item.get("semantic") is semantic, f"{group}.semantic invariant violated", errors)
        _expect(
            item.get("action_containment") is containment,
            f"{group}.action_containment invariant violated",
            errors,
        )

    evaluation = config.get("evaluation", {})
    _expect(evaluation.get("offline_first") is True, "offline_first must be true", errors)
    _expect(
        evaluation.get("bootstrap_unit") == "episode_cluster",
        "bootstrap_unit must be episode_cluster",
        errors,
    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--allow-template", action="store_true")
    args = parser.parse_args()

    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"CONFIG_READ_ERROR: {exc}")
        return 2

    errors = validate_config(config, allow_template=args.allow_template)
    if errors:
        print("VLA_CONFIG_INVALID")
        for error in errors:
            print(f"- {error}")
        return 1
    print("VLA_CONFIG_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
