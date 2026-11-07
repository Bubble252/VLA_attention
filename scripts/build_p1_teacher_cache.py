#!/usr/bin/env python3
"""Build P1-B A10 teacher labels on the frozen C1 training cache.

The teacher is the P0 A10 privileged lateral controller.  Labels are computed
only for training episodes; ``white_right`` is rejected.  The output is aligned
to ``cache_train_v2`` by ``source_index`` so P1-C can reuse the existing route,
geometry, proprio and base-action arrays without copying 4096-D features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_e1_failure_atlas import gripper_width, observation_positions  # noqa: E402
from scripts.k4_lite_t4_common import load_t4_cache  # noqa: E402
from src.vla_attention.t4b_v8_control import ArmState, apply_arm_control  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def is_white_right(language: Any) -> bool:
    text = str(language).lower()
    return "white mug" in text and "right plate" in text


def prompt_for_language(language: str) -> str:
    text = str(language).lower()
    if "red mug" in text:
        source = "red"
    elif "white mug" in text:
        source = "white"
    else:
        raise ValueError(f"unrecognized source object in {language!r}")
    if "left plate" in text:
        target = "left"
    elif "right plate" in text:
        target = "right"
    else:
        raise ValueError(f"unrecognized target plate in {language!r}")
    return f"{source}_{target}"


def prompt_metadata(manifest: Path) -> dict[str, dict[str, Any]]:
    by_prompt: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for row in read_jsonl(manifest):
        prompt = str(row["prompt_id"])
        by_prompt[prompt].add(
            (
                str(row["bddl_source_path"]),
                str(row["target_object"]),
                str(row["target_receptacle"]),
            )
        )
    result: dict[str, dict[str, Any]] = {}
    for prompt, values in by_prompt.items():
        if len(values) != 1:
            raise ValueError(f"ambiguous prompt metadata for {prompt}: {sorted(values)}")
        bddl, target, receptacle = next(iter(values))
        result[prompt] = {
            "bddl": bddl,
            "target_object": target,
            "target_receptacle": receptacle,
        }
    return result


def load_demo_states(path: Path, demo_key: str) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as handle:
        if demo_key not in handle["data"]:
            raise KeyError(f"{demo_key} not found in {path}")
        return np.asarray(handle["data"][demo_key]["states"], dtype=np.float64)


def load_inventory(path: Path, split: str) -> dict[str, Any]:
    payload = read_json(path)
    if split not in ("train", "validation"):
        raise ValueError(f"unsupported inventory split: {split}")
    rows = payload["splits"][split]
    expected = 96 if split == "train" else 15
    if len(rows) != expected:
        raise ValueError(f"expected {expected} {split} episodes, got {len(rows)}")
    if any(is_white_right(row.get("language")) for row in rows):
        raise ValueError(f"white_right found in P1-B {split} inventory")
    return payload


def validate_cache_alignment(
    cache_meta: list[dict[str, Any]], inventory: dict[str, Any], split: str
) -> dict[str, list[dict[str, Any]]]:
    inventory_by_id = {str(row["episode_id"]): row for row in inventory["splits"][split]}
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cache_meta:
        episode_id = str(row["episode_id"])
        if episode_id not in inventory_by_id:
            raise ValueError(f"cache episode missing from {split} inventory: {episode_id}")
        by_episode[episode_id].append(row)
    if set(by_episode) != set(inventory_by_id):
        missing = sorted(set(inventory_by_id) - set(by_episode))
        raise ValueError(f"{split} episodes missing from cache: {missing}")
    for episode_id, rows in by_episode.items():
        rows.sort(key=lambda item: int(item["timestep"]))
        timesteps = [int(row["timestep"]) for row in rows]
        if timesteps != list(range(len(timesteps))):
            raise ValueError(f"non-contiguous timesteps for {episode_id}")
        inventory_row = inventory_by_id[episode_id]
        span = int(inventory_row["hdf5_action_end"]) - int(inventory_row["hdf5_action_start"])
        if len(timesteps) > span:
            raise ValueError(f"cache timesteps exceed HDF5 span for {episode_id}")
    return by_episode


def build_env(bddl_path: str, resolution: int):
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(0)
    env.reset()
    return env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    cache = load_t4_cache(args.cache)
    arrays = cache["arrays"]
    cache_meta = cache["meta"]
    inventory = load_inventory(args.inventory, args.split)
    episode_rows = validate_cache_alignment(cache_meta, inventory, args.split)
    prompts = prompt_metadata(args.manifest)
    config = read_json(args.config)
    metrics = config["metrics"]
    arm_cfg = config["arms"]["A10"]
    if float(arm_cfg["fine_cap"]) != 0.10:
        raise ValueError("A10 teacher must keep fine_cap=0.10")
    if float(arm_cfg["release_xy_m"]) != 0.03 or float(arm_cfg["release_z_m"]) != 0.05:
        raise ValueError("A10 release thresholds drifted")

    inventory_by_id = {str(row["episode_id"]): row for row in inventory["splits"][args.split]}
    episode_ids = sorted(episode_rows)
    if args.max_episodes:
        episode_ids = episode_ids[: args.max_episodes]

    envs: dict[str, Any] = {}
    label_rows: list[dict[str, Any]] = []
    source_indices: list[int] = []
    teacher_delta: list[np.ndarray] = []
    teacher_delta_chunk: list[np.ndarray] = []
    teacher_action_xy: list[np.ndarray] = []
    teacher_release: list[int] = []
    teacher_release_active: list[int] = []
    teacher_xy_m: list[float] = []
    teacher_z_m: list[float] = []
    errors: list[dict[str, Any]] = []

    try:
        for episode_id in episode_ids:
            row = inventory_by_id[episode_id]
            task_name = str(row["task_name"])
            language = str(row["language"])
            prompt = prompt_for_language(language)
            if prompt not in prompts:
                raise KeyError(f"missing prompt metadata for {prompt}")
            meta = prompts[prompt]
            if task_name not in envs:
                envs[task_name] = build_env(meta["bddl"], args.resolution)
            env = envs[task_name]
            states = load_demo_states(
                Path(inventory["tasks"][task_name]["hdf5"]), str(row["demo_key"])
            )
            start = int(row["hdf5_action_start"])
            end = int(row["hdf5_action_end"])
            if end > len(states):
                raise ValueError(f"HDF5 span exceeds demo states for {episode_id}")
            arm_state = ArmState()
            baseline_z = None
            for item in episode_rows[episode_id]:
                timestep = int(item["timestep"])
                source_index = int(item["index"])
                state = states[start + timestep].copy()
                state[0] = 0.0
                obs = env.set_init_state(state)
                env.sim.forward()
                objects = observation_positions(obs)
                target = str(meta["target_object"])
                plate = str(meta["target_receptacle"])
                if target not in objects or plate not in objects:
                    raise KeyError(f"missing object/plate for {episode_id} at t={timestep}")
                if baseline_z is None:
                    baseline_z = float(objects[target][2])
                base_action = np.asarray(arrays["base_action"][source_index], dtype=np.float32)
                teacher_action, telemetry, arm_state = apply_arm_control(
                    base_action[0],
                    objects=objects,
                    gripper_width=gripper_width(obs),
                    target=target,
                    plate=plate,
                    baseline_z=baseline_z,
                    control_step=timestep,
                    arm_cfg=arm_cfg,
                    metrics_cfg=metrics,
                    state=arm_state,
                )
                cap = float(arm_cfg["fine_cap"])
                delta = np.clip(
                    np.asarray(teacher_action[:2], dtype=np.float64)
                    - np.asarray(base_action[0, :2], dtype=np.float64),
                    -cap,
                    cap,
                )
                delta_chunk = np.clip(
                    np.asarray(teacher_action[None, :2], dtype=np.float64)
                    - np.asarray(base_action[:, :2], dtype=np.float64),
                    -cap,
                    cap,
                )
                release_active = bool(telemetry["release_active"])
                if release_active:
                    delta[:] = 0.0
                    delta_chunk[:] = 0.0
                source_indices.append(source_index)
                teacher_delta.append(delta.astype(np.float32))
                teacher_delta_chunk.append(delta_chunk.astype(np.float32))
                teacher_action_xy.append(
                    np.asarray(teacher_action[:2], dtype=np.float32)
                )
                teacher_release.append(int(bool(telemetry["release_triggered"])))
                teacher_release_active.append(int(release_active))
                teacher_xy_m.append(float(telemetry["xy_m"]))
                teacher_z_m.append(float(telemetry["z_m"]))
                label_rows.append(
                    {
                        "source_index": source_index,
                        "episode_id": episode_id,
                        "timestep": timestep,
                        "task_name": task_name,
                        "language": language,
                        "prompt": prompt,
                        "demo_key": row["demo_key"],
                        "state_index": start + timestep,
                        "split": args.split,
                        "target_object": target,
                        "target_receptacle": plate,
                        "teacher_applied": telemetry["applied"],
                        "teacher_release": int(bool(telemetry["release_triggered"])),
                        "teacher_release_active": int(release_active),
                        "teacher_cap_mode": telemetry["cap_mode"],
                    }
                )
    except Exception as exc:
        errors.append({"error": repr(exc), "episode_id": episode_id, "timestep": timestep})
        raise
    finally:
        for env in envs.values():
            env.close()

    source_index_array = np.asarray(source_indices, dtype=np.int32)
    delta_array = np.asarray(teacher_delta, dtype=np.float32)
    delta_chunk_array = np.asarray(teacher_delta_chunk, dtype=np.float32)
    action_xy_array = np.asarray(teacher_action_xy, dtype=np.float32)
    release_array = np.asarray(teacher_release, dtype=np.uint8)
    release_active_array = np.asarray(teacher_release_active, dtype=np.uint8)
    xy_m_array = np.asarray(teacher_xy_m, dtype=np.float32)
    z_m_array = np.asarray(teacher_z_m, dtype=np.float32)
    delta_norm = np.linalg.norm(delta_array, axis=1)
    bins = np.digitize(delta_norm, [0.01, 0.03, 0.05, 0.10])
    release_episodes = len(
        {row["episode_id"] for row in label_rows if int(row["teacher_release"]) == 1}
    )
    finite = all(
        np.isfinite(value).all()
        for value in (
            source_index_array,
            delta_array,
            delta_chunk_array,
            action_xy_array,
            xy_m_array,
            z_m_array,
        )
    )
    episode_check_name = (
        "all_train_episodes_present"
        if args.split == "train"
        else "all_validation_episodes_present"
    )
    white_check_name = (
        "white_right_train_zero"
        if args.split == "train"
        else "white_right_validation_zero"
    )
    checks = {
        episode_check_name: len({row["episode_id"] for row in label_rows})
        == len(episode_ids),
        "transition_count_matches": len(label_rows) == int(arrays["base_action"].shape[0])
        if not args.max_episodes
        else True,
        white_check_name: not any(is_white_right(row["language"]) for row in label_rows),
        "plateau_bucket_nonempty": int(((delta_norm >= 0.027) & (delta_norm <= 0.035)).sum()) > 0,
        "large_bucket_nonempty": int(((delta_norm >= 0.05) & (delta_norm <= 0.10)).sum()) > 0,
        "delta_xy_abs_max_le_010": float(np.abs(delta_array).max()) <= 0.10 + 1e-6,
        "all_arrays_finite": bool(finite),
        "no_generation_errors": not errors,
    }
    if args.split == "train":
        checks["release_positive_transitions_ge_50"] = int(release_array.sum()) >= 50
        checks["release_positive_episodes_ge_50"] = release_episodes >= 50
    else:
        checks["release_positive_transitions_gt_0"] = int(release_array.sum()) > 0
        checks["release_positive_episodes_gt_0"] = release_episodes > 0
    status = "PASS" if all(checks.values()) else "INVALID"
    np.save(args.output / "source_index.npy", source_index_array)
    np.save(args.output / "teacher_delta_xy.npy", delta_array)
    np.save(args.output / "teacher_delta_chunk.npy", delta_chunk_array)
    np.save(args.output / "teacher_action_xy.npy", action_xy_array)
    np.save(args.output / "teacher_release.npy", release_array)
    np.save(args.output / "teacher_release_active.npy", release_active_array)
    np.save(args.output / "teacher_xy_m.npy", xy_m_array)
    np.save(args.output / "teacher_z_m.npy", z_m_array)
    (args.output / "meta.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in label_rows)
    )
    summary = {
        "status": status,
        "run_contract_status": status,
        "cache": str(args.cache),
        "inventory": str(args.inventory),
        "inventory_split": args.split,
        "inventory_sha256": sha256_file(args.inventory),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "teacher_arm": "A10",
        "teacher_fine_cap": float(arm_cfg["fine_cap"]),
        "teacher_release_xy_m": float(arm_cfg["release_xy_m"]),
        "teacher_release_z_m": float(arm_cfg["release_z_m"]),
        "episodes": len({row["episode_id"] for row in label_rows}),
        "transitions": len(label_rows),
        "languages": dict(sorted(Counter(row["language"] for row in label_rows).items())),
        "release_positive_transitions": int(release_array.sum()),
        "release_positive_episodes": release_episodes,
        "release_active_transitions": int(release_active_array.sum()),
        "delta_norm_quantiles": {
            f"p{int(point * 100):02d}": float(np.quantile(delta_norm, point))
            for point in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
        },
        "delta_bins": {
            "thresholds_m": [0.01, 0.03, 0.05, 0.10],
            "counts": np.bincount(bins, minlength=5).tolist(),
            "fractions": (np.bincount(bins, minlength=5) / max(len(label_rows), 1)).tolist(),
            "plateau_count": int(((delta_norm >= 0.027) & (delta_norm <= 0.035)).sum()),
            "large_count": int(((delta_norm >= 0.05) & (delta_norm <= 0.10)).sum()),
        },
        "gates": checks,
        "errors": errors,
        "arrays": {
            "source_index": list(source_index_array.shape),
            "teacher_delta_xy": list(delta_array.shape),
            "teacher_delta_chunk": list(delta_chunk_array.shape),
            "teacher_action_xy": list(action_xy_array.shape),
            "teacher_release": list(release_array.shape),
            "teacher_release_active": list(release_active_array.shape),
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
