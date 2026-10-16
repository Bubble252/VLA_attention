"""Read-only LIBERO task/data audit for instruction-evidence supervision."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import shlex
import subprocess
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
REPO = Path("/home/bubble/类脑计算/VLM终局/references/repos/libero")
LIBERO = REPO / "libero/libero"
TASK_MAP_PATH = LIBERO / "benchmark/libero_suite_task_map.py"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def task_map() -> dict[str, list[str]]:
    tree = ast.parse(TASK_MAP_PATH.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "libero_task_map" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("libero_task_map not found")


def balanced_section(text: str, name: str) -> str:
    start = text.find(f"(:{name}")
    if start < 0:
        return ""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError(f"unbalanced section {name}")


def section_atoms(section: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_+.-]+", section)


def scene_signature(text: str) -> str:
    for name in ("language", "obj_of_interest", "goal"):
        part = balanced_section(text, name)
        text = text.replace(part, "")
    normalized = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()


def find_init(suite: str, name: str) -> Path | None:
    base = LIBERO / "init_files" / suite
    for suffix in (".pruned_init", ".init"):
        p = base / f"{name}{suffix}"
        if p.exists():
            return p
    return None


def load_init(path: Path | None):
    if path is None:
        return None
    return torch.as_tensor(torch.load(path, map_location="cpu", weights_only=False)).float()


def remote_demo_counts() -> dict:
    remote = "/vepfs-mlp2/c20250405/400040/transfer/vla_attention/vla_workspace/artifacts/rlds_inventory_v2_20260928/inventory.jsonl"
    code = (
        "import json,collections;"
        f"p={remote!r};"
        "r=[json.loads(x) for x in open(p)];"
        "print(json.dumps({'path':p,'episodes':len(r),'tasks':collections.Counter(x['task_name'] for x in r),"
        "'action_shapes':sorted({tuple(x['action_shape'][1:]) for x in r}),"
        "'camera_shapes':r[0]['camera_shapes'],'data_revisions':sorted({x['data_revision'] for x in r})}))"
    )
    remote_command = "python3 -c " + shlex.quote(code)
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "vla101", remote_command],
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


PAIR_NAMES = [
    ("libero_goal", "put_the_bowl_on_the_plate", "put_the_bowl_on_the_stove", "target/container swap"),
    ("libero_goal", "put_the_bowl_on_the_plate", "put_the_bowl_on_top_of_the_cabinet", "target/container swap"),
    ("libero_90", "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate", "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate", "target plate swap"),
    ("libero_90", "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate", "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate", "source mug swap"),
    ("libero_90", "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate", "LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate", "source mug swap"),
    ("libero_90", "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy", "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy", "target compartment swap"),
    ("libero_90", "KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate", "KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet", "target/container swap"),
    ("libero_90", "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_left_of_the_plate", "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate", "target relation swap"),
    ("libero_90", "LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray", "LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray", "source/target role swap"),
    ("libero_90", "KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate", "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate", "source position variant"),
]


def main() -> None:
    outputs = [ROOT / "libero_task_inventory.json", ROOT / "candidate_task_pairs.csv"]
    if any(p.exists() for p in outputs):
        raise SystemExit("Refusing to overwrite audit outputs")
    mapping = task_map()
    demo = remote_demo_counts()
    records = []
    arrays = {}
    for suite, names in mapping.items():
        for task_id, name in enumerate(names):
            bddl = LIBERO / "bddl_files" / suite / f"{name}.bddl"
            text = bddl.read_text()
            init = find_init(suite, name)
            array = load_init(init)
            arrays[(suite, name)] = array
            language = balanced_section(text, "language")
            language = re.sub(r"^\(:language\s*", "", language).rstrip(")").strip()
            objects = section_atoms(balanced_section(text, "objects"))[1:]
            interest = section_atoms(balanced_section(text, "obj_of_interest"))[1:]
            init_atoms = section_atoms(balanced_section(text, "init"))[1:]
            goal_atoms = section_atoms(balanced_section(text, "goal"))[1:]
            count = int(demo["tasks"].get(name, 0))
            records.append(
                {
                    "suite": suite,
                    "task_id": task_id,
                    "task_name": name,
                    "language": language,
                    "bddl_path": str(bddl),
                    "bddl_sha256": sha256(bddl),
                    "scene_signature_without_language_interest_goal": scene_signature(text),
                    "objects_tokens": objects,
                    "objects_of_interest": interest,
                    "init_predicate_tokens": init_atoms,
                    "goal_tokens": goal_atoms,
                    "init_path": str(init) if init else None,
                    "init_sha256": sha256(init) if init else None,
                    "init_shape": list(array.shape) if array is not None else None,
                    "current_remote_demo_count": count,
                    "current_remote_demo_source": demo["path"] if count else None,
                    "current_remote_action_tail_shape": demo["action_shapes"] if count else None,
                    "current_remote_camera_shapes": demo["camera_shapes"] if count else None,
                }
            )
    by_key = {(r["suite"], r["task_name"]): r for r in records}
    fields = [
        "pair_id", "suite", "task_a_id", "task_b_id", "task_a", "task_b", "instruction_a", "instruction_b",
        "pair_type", "objects_interest_a", "objects_interest_b", "scene_structure_match", "init_shape_match",
        "exact_shared_init_state", "minimum_init_state_l2", "current_demo_count_a", "current_demo_count_b",
        "camera_config", "action_schema", "legal_expert_actions_current", "constructibility", "reason",
    ]
    rows = []
    for pair_id, (suite, a_name, b_name, kind) in enumerate(PAIR_NAMES, 1):
        a, b = by_key[(suite, a_name)], by_key[(suite, b_name)]
        aa, bb = arrays[(suite, a_name)], arrays[(suite, b_name)]
        shape_match = aa is not None and bb is not None and tuple(aa.shape[1:]) == tuple(bb.shape[1:])
        if shape_match:
            distances = torch.cdist(aa, bb)
            exact = bool((distances == 0).any())
            min_l2 = float(distances.min())
        else:
            exact, min_l2 = False, None
        demos = a["current_remote_demo_count"] > 0 and b["current_remote_demo_count"] > 0
        strict_trainable = demos and exact
        if strict_trainable:
            status = "Trainable paired supervision"
            reason = "Both tasks have legal demonstrations and share an exact audited init state."
        else:
            status = "Diagnostic-only intervention"
            reason = "Scene family is useful, but current data lacks exact-state legal expert actions for both targets."
        rows.append(
            {
                "pair_id": pair_id,
                "suite": suite,
                "task_a_id": a["task_id"],
                "task_b_id": b["task_id"],
                "task_a": a_name,
                "task_b": b_name,
                "instruction_a": a["language"],
                "instruction_b": b["language"],
                "pair_type": kind,
                "objects_interest_a": "|".join(a["objects_of_interest"]),
                "objects_interest_b": "|".join(b["objects_of_interest"]),
                "scene_structure_match": a["scene_signature_without_language_interest_goal"] == b["scene_signature_without_language_interest_goal"],
                "init_shape_match": shape_match,
                "exact_shared_init_state": exact,
                "minimum_init_state_l2": min_l2,
                "current_demo_count_a": a["current_remote_demo_count"],
                "current_demo_count_b": b["current_remote_demo_count"],
                "camera_config": json.dumps(a["current_remote_camera_shapes"], ensure_ascii=False) if demos else "not currently inventoried",
                "action_schema": "7D delta EEF in current RLDS" if demos else "official raw dataset not currently inventoried",
                "legal_expert_actions_current": demos,
                "constructibility": status,
                "reason": reason,
            }
        )
    inventory = {
        "libero_git_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "task_map_path": str(TASK_MAP_PATH),
        "task_map_sha256": sha256(TASK_MAP_PATH),
        "task_count": len(records),
        "suite_counts": dict(Counter(r["suite"] for r in records)),
        "current_remote_inventory": demo,
        "tasks": records,
        "task0_assessment": {
            "task": by_key[("libero_spatial", mapping["libero_spatial"][0])],
            "has_two_black_bowls": True,
            "instruction_selects_one_by_spatial_relation": True,
            "paired_alternative_instruction_with_legal_action_in_current_data": False,
            "verdict": "Useful for spatial-reference diagnosis; insufficient alone for target-retargeting training.",
        },
    }
    outputs[0].write_text(json.dumps(inventory, ensure_ascii=False, indent=2))
    with outputs[1].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"tasks": len(records), "pairs": len(rows), "remote_episodes": demo["episodes"]}, indent=2))


if __name__ == "__main__":
    main()
