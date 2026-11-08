#!/usr/bin/env python3
"""Build the Goal16 R-PRED grounding sidecar from frozen Qwen + point head.

The point head is queried once per frozen ``white_right`` init on the same two
224x224 images used by the policy.  Its destination point is mapped back to the
table plane with the fixed LIBERO agent-view camera calibration.  No simulator
pose is used by the predictor; oracle poses are written only for audit and for
computing the pre-registered grounding error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_POINT_HEAD_SHA256 = (
    "693f0dd8086a4d64cd4f1064a7f4d5c1f934f0914123df5478cdc1f86815fce7"
)
EXPECTED_MANIFEST_SHA256 = (
    "e9564ca9aef800e618e3fdb5d144892eabe9471815bbd24bae3140583946f45e"
)
CALIBRATION_INIT_IDS = frozenset(
    {
        "init_0e94a4e6b9622c6b",
        "init_2edf49999eb35566",
        "init_3500df19a4d3b5a1",
    }
)
EXPECTED_INIT_COUNT = 15
EXPECTED_CONFIRMATION_COUNT = 12
IMAGE_TOKEN_INDEX = 151655


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def select_white_right(rows: Iterable[dict[str, Any]], max_inits: int) -> list[dict[str, Any]]:
    selected = sorted(
        [dict(row) for row in rows if str(row.get("prompt_id")) == "white_right"],
        key=lambda row: str(row["shared_init_id"]),
    )
    if max_inits > 0:
        selected = selected[:max_inits]
    if not selected:
        raise ValueError("no white_right rows selected")
    if len({str(row["shared_init_id"]) for row in selected}) != len(selected):
        raise ValueError("duplicate shared_init_id in selected white_right rows")
    return selected


def _image_token_counts(image_grid_thw: Any, merge_size: int) -> list[int]:
    counts: list[int] = []
    for grid in image_grid_thw.detach().cpu().tolist():
        if len(grid) != 3:
            raise ValueError(f"expected image grid [t,h,w], got {grid}")
        t, height, width = (int(value) for value in grid)
        divisor = merge_size * merge_size
        if t * height * width % divisor:
            raise ValueError(f"grid {grid} is not divisible by merge_size^2")
        counts.append((t * height * width) // divisor)
    return counts


def _assign_view_ids(visual_count: int, counts: list[int], device: Any) -> Any:
    import torch

    if len(counts) != 2 or sum(counts) != visual_count:
        if visual_count % 2:
            raise ValueError(
                f"visual token count {visual_count} does not match two views {counts}"
            )
        counts = [visual_count // 2, visual_count // 2]
    return torch.cat(
        [
            torch.full((count,), view, dtype=torch.long, device=device)
            for view, count in enumerate(counts)
        ]
    )


def _load_point_head(path: Path, device: Any) -> Any:
    import torch
    from starVLA.model.modules.point_head import PointQueryHead

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    state = payload.get("final_model", payload.get("model", payload)) if isinstance(payload, dict) else payload
    args = payload.get("args", {}) if isinstance(payload, dict) else {}
    if "query_embed" not in state:
        raise ValueError(f"point head state has no query_embed: {sorted(state)}")
    hidden_dim = int(state["query_embed"].shape[1])
    head = PointQueryHead(
        hidden_dim=hidden_dim,
        num_heads=int(args.get("num_heads", 8)),
        num_layers=int(args.get("num_layers", 2)),
        dropout=float(args.get("dropout", 0.0)),
    )
    head.load_state_dict(state, strict=True)
    head.to(device=device)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad = False
    return head


def _camera_matrices(sim: Any) -> tuple[np.ndarray, np.ndarray]:
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
    )

    intrinsic = get_camera_intrinsic_matrix(sim, "agentview", 256, 256)
    extrinsic = get_camera_extrinsic_matrix(sim, "agentview")
    return np.asarray(intrinsic, dtype=np.float64), np.asarray(extrinsic, dtype=np.float64)


def _unproject_table_plane(
    pixel_uv: tuple[float, float],
    *,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    table_z_m: float,
) -> np.ndarray:
    u, v = (float(value) for value in pixel_uv)
    ray_camera = np.linalg.inv(intrinsic) @ np.array([u, v, 1.0], dtype=np.float64)
    ray_world = extrinsic[:3, :3] @ ray_camera
    if abs(float(ray_world[2])) < 1e-9:
        raise ValueError("camera ray is parallel to the destination plane")
    camera_center = extrinsic[:3, 3]
    scale = (float(table_z_m) - float(camera_center[2])) / float(ray_world[2])
    if scale <= 0.0:
        raise ValueError("destination plane is behind the camera")
    return camera_center + scale * ray_world


def _project_point(
    point_xyz: Iterable[float],
    *,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> tuple[float, float]:
    point = np.asarray(list(point_xyz), dtype=np.float64)
    camera = np.linalg.inv(extrinsic) @ np.r_[point, 1.0]
    # robosuite's agentview extrinsic stores a camera frame whose +z axis
    # points into the scene (CV convention), unlike OpenGL camera frames.
    if camera[2] <= 1e-9:
        raise ValueError("point is behind the camera")
    projected = intrinsic @ camera[:3]
    return float(projected[0] / projected[2]), float(projected[1] / projected[2])


def _point_head_predictions(
    *,
    interface: Any,
    point_head: Any,
    agentview: Image.Image,
    wrist: Image.Image,
    instruction: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    inputs = interface.build_qwenvl_inputs(
        images=[[agentview, wrist]],
        instructions=[instruction],
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = interface(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
    hidden = outputs.hidden_states[-1][0]
    input_ids = inputs["input_ids"][0]
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask[0].bool()
    image_mask = input_ids == IMAGE_TOKEN_INDEX
    visual_positions = image_mask.nonzero(as_tuple=False).flatten()
    language_positions = (attention_mask & ~image_mask).nonzero(as_tuple=False).flatten()
    if visual_positions.numel() == 0 or language_positions.numel() == 0:
        raise ValueError("missing visual or language tokens")
    grid = inputs.get("image_grid_thw")
    if grid is None:
        raise ValueError("Qwen input has no image_grid_thw")
    merge_size = int(interface.model.config.vision_config.spatial_merge_size)
    counts = _image_token_counts(grid, merge_size)
    visual_view_ids = _assign_view_ids(
        int(visual_positions.numel()), counts, hidden.device
    )

    head_dtype = next(point_head.parameters()).dtype
    visual_tokens = hidden[visual_positions].unsqueeze(0).to(dtype=head_dtype)
    language_tokens = hidden[language_positions].unsqueeze(0).to(dtype=head_dtype)
    visual_mask = torch.ones(
        (1, visual_tokens.shape[1]), dtype=torch.bool, device=hidden.device
    )
    language_mask = torch.ones(
        (1, language_tokens.shape[1]), dtype=torch.bool, device=hidden.device
    )
    with torch.inference_mode():
        points, _ = point_head(
            visual_tokens,
            visual_mask,
            language_tokens,
            language_mask,
            visual_view_ids=visual_view_ids.unsqueeze(0),
        )
    points_np = points[0].detach().float().cpu().numpy()
    if points_np.shape != (2, 2) or not np.isfinite(points_np).all():
        raise ValueError(f"invalid point head output: {points_np!r}")
    return points_np, {
        "visual_token_count": int(visual_positions.numel()),
        "language_token_count": int(language_positions.numel()),
        "visual_token_counts_by_view": counts,
        "head_dtype": str(head_dtype),
    }


def _draw_overlay(
    agentview: np.ndarray,
    *,
    object_xy: tuple[float, float],
    destination_xy: tuple[float, float],
    oracle_xy: tuple[float, float] | None,
    output: Path,
) -> None:
    image = Image.fromarray(np.asarray(agentview, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)

    def circle(point: tuple[float, float], color: tuple[int, int, int]) -> None:
        x, y = float(point[0]), float(point[1])
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=color, width=2)

    circle(object_xy, (255, 40, 40))
    circle(destination_xy, (40, 80, 255))
    if oracle_xy is not None:
        circle(oracle_xy, (40, 200, 40))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--point-head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", default=EXPECTED_MANIFEST_SHA256)
    parser.add_argument("--expected-point-head-sha256", default=EXPECTED_POINT_HEAD_SHA256)
    parser.add_argument("--max-inits", type=int, default=0)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--table-z-m", type=float, default=0.43927283)
    parser.add_argument("--max-destination-error-m", type=float, default=0.15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # LIBERO imports TensorFlow at module scope.  Hide the GPUs before that
    # import so TF cannot reserve a device that the policy/point head need.
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    import torch
    from omegaconf import OmegaConf
    from libero.libero.envs import OffScreenRenderEnv
    import experiments.robot.libero.run_libero_eval as official
    from starVLA.model.modules.vlm.QWen2_5 import _QWen_VL_Interface

    manifest_sha = sha256_file(args.manifest)
    point_head_sha = sha256_file(args.point_head)
    if manifest_sha != args.expected_manifest_sha256:
        raise ValueError(f"manifest SHA mismatch: {manifest_sha}")
    if point_head_sha != args.expected_point_head_sha256:
        raise ValueError(f"point head SHA mismatch: {point_head_sha}")

    rows = read_manifest(args.manifest)
    selected = select_white_right(rows, args.max_inits)
    if not args.allow_partial and len(selected) != EXPECTED_INIT_COUNT:
        raise ValueError(f"expected {EXPECTED_INIT_COUNT} white_right inits, got {len(selected)}")
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device("cuda")
    config = OmegaConf.load(args.config)
    interface = _QWen_VL_Interface(config=config)
    interface.model.to(device)
    interface.model.eval()
    point_head = _load_point_head(args.point_head, device)

    image_root = args.output / "images"
    overlay_root = args.output / "overlays"
    image_root.mkdir(parents=True, exist_ok=False)
    overlay_root.mkdir(parents=True, exist_ok=False)
    rows_out: list[dict[str, Any]] = []
    env = None
    camera: dict[str, Any] | None = None

    try:
        for index, row in enumerate(selected, start=1):
            if env is None:
                env = OffScreenRenderEnv(
                    bddl_file_name=str(row["bddl_path"]),
                    camera_heights=int(args.resolution),
                    camera_widths=int(args.resolution),
                )
                env.seed(0)
            env.reset()
            init_state = np.asarray(row["init_state"], dtype=np.float64).copy()
            if row.get("sim_time_set_to_zero", False):
                init_state[0] = 0.0
            obs = env.set_init_state(init_state)
            for _ in range(int(args.num_steps_wait)):
                obs, _, _, _ = env.step(official.get_libero_dummy_action("openvla"))

            observation, _raw = official.prepare_observation(obs, int(args.image_size))
            agentview = Image.fromarray(observation["full_image"]).convert("RGB")
            wrist = Image.fromarray(observation["wrist_image"]).convert("RGB")
            init_id = str(row["shared_init_id"])
            agentview_path = image_root / f"{init_id}__agentview.png"
            wrist_path = image_root / f"{init_id}__wrist.png"
            agentview.save(agentview_path)
            wrist.save(wrist_path)

            if camera is None:
                intrinsic, extrinsic = _camera_matrices(env.env.sim)
                camera = {
                    "intrinsic_K": intrinsic.tolist(),
                    "camera_to_env": extrinsic.tolist(),
                    "shape": [int(args.resolution), int(args.resolution)],
                    "table_z_m": float(args.table_z_m),
                }
            else:
                intrinsic = np.asarray(camera["intrinsic_K"], dtype=np.float64)
                extrinsic = np.asarray(camera["camera_to_env"], dtype=np.float64)

            points, diagnostics = _point_head_predictions(
                interface=interface,
                point_head=point_head,
                agentview=agentview,
                wrist=wrist,
                instruction=str(row["prompt_language"]),
            )
            pixel_scale = float(args.image_size - 1)
            object_px = points[0] * pixel_scale
            destination_px = points[1] * pixel_scale
            scale = float(args.resolution) / float(args.image_size)
            predicted_xyz = _unproject_table_plane(
                (float(destination_px[0]) * scale, float(destination_px[1]) * scale),
                intrinsic=intrinsic,
                extrinsic=extrinsic,
                table_z_m=float(args.table_z_m),
            )
            oracle_xyz = np.asarray(
                obs[str(row["target_receptacle"]) + "_pos"], dtype=np.float64
            )
            oracle_px_raw = _project_point(
                oracle_xyz,
                intrinsic=intrinsic,
                extrinsic=extrinsic,
            )
            oracle_px = np.asarray(oracle_px_raw, dtype=np.float64) / scale
            destination_error_m = float(
                np.linalg.norm(predicted_xyz[:2] - oracle_xyz[:2])
            )
            destination_pixel_error = float(
                np.linalg.norm(destination_px - oracle_px)
            )
            overlay_path = overlay_root / f"{init_id}.png"
            _draw_overlay(
                observation["full_image"],
                object_xy=tuple(float(value) for value in object_px),
                destination_xy=tuple(float(value) for value in destination_px),
                oracle_xy=tuple(float(value) for value in oracle_px),
                output=overlay_path,
            )
            rows_out.append(
                {
                    "shared_init_id": init_id,
                    "row_id": row["row_id"],
                    "split": "calibration" if init_id in CALIBRATION_INIT_IDS else "confirmation",
                    "prompt_id": row["prompt_id"],
                    "prompt_language": row["prompt_language"],
                    "target_object": row.get("target_object"),
                    "target_receptacle": row.get("target_receptacle"),
                    "agentview_path": str(agentview_path),
                    "wrist_path": str(wrist_path),
                    "agentview_sha256": sha256_file(agentview_path),
                    "wrist_sha256": sha256_file(wrist_path),
                    "overlay_path": str(overlay_path),
                    "object_point_norm": [float(value) for value in points[0]],
                    "destination_point_norm": [float(value) for value in points[1]],
                    "object_point_px": [float(value) for value in object_px],
                    "destination_point_px": [float(value) for value in destination_px],
                    "oracle_destination_point_px": [float(value) for value in oracle_px],
                    "destination_position_xyz": predicted_xyz.tolist(),
                    "oracle_destination_xyz": oracle_xyz.tolist(),
                    "destination_error_m": destination_error_m,
                    "destination_pixel_error": destination_pixel_error,
                    "grounding_miss": bool(destination_error_m > float(args.max_destination_error_m)),
                    **diagnostics,
                }
            )
            print(
                json.dumps(
                    {
                        "processed": index,
                        "total": len(selected),
                        "shared_init_id": init_id,
                        "destination_error_m": destination_error_m,
                        "grounding_miss": rows_out[-1]["grounding_miss"],
                    }
                ),
                flush=True,
            )
    finally:
        if env is not None:
            env.close()

    sidecar = args.output / "predicted_points.jsonl"
    with sidecar.open("w") as handle:
        for row in rows_out:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    errors = [float(row["destination_error_m"]) for row in rows_out]
    summary = {
        "schema": "goal16-grounding-sidecar-v1",
        "status": "complete",
        "config": str(args.config),
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_sha,
        "point_head": str(args.point_head),
        "point_head_sha256": point_head_sha,
        "sidecar": str(sidecar),
        "sidecar_sha256": sha256_file(sidecar),
        "camera": camera,
        "selected_init_count": len(selected),
        "calibration_init_count": sum(
            row["shared_init_id"] in CALIBRATION_INIT_IDS for row in selected
        ),
        "confirmation_init_count": sum(
            row["shared_init_id"] not in CALIBRATION_INIT_IDS for row in selected
        ),
        "grounding_miss_count": sum(bool(row["grounding_miss"]) for row in rows_out),
        "max_destination_error_m": max(errors) if errors else None,
        "mean_destination_error_m": float(np.mean(errors)) if errors else None,
        "max_allowed_destination_error_m": float(args.max_destination_error_m),
        "passed": bool(errors and max(errors) <= float(args.max_destination_error_m)),
        "notes": [
            "Point head inputs are the two policy images; no mask/object id/success/future frame is used.",
            "The predicted destination pixel is unprojected with a fixed agent-view camera calibration.",
            "Oracle destination is recorded only for audit and grounding-error measurement.",
        ],
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
