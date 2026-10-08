"""Export LIBERO RLDS frames and source/target phrase rows for semantic cache.

This is the general held-out counterpart of export_libero_cache_inputs.py. It
only creates PNGs and JSONL manifests; it does not run a teacher model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    import tensorflow as tf
    from PIL import Image

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", required=True, choices=("train", "offline_eval", "eval"))
    parser.add_argument("--source-phrase", default="the black bowl")
    parser.add_argument("--target-phrase", default="the plate")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    frames = []
    semantic = []
    for row in rows:
        if row["task_name"] != "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate":
            raise ValueError("Task phrase mapping has not been specified for this task")
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        dataset = tf.data.TFRecordDataset([row["source_shard"]], num_parallel_reads=1).with_options(options)
        raw = next(iter(dataset.skip(row["record_index"]).take(1))).numpy()
        if hashlib.sha256(raw).hexdigest() != row["episode_id"]:
            raise ValueError("Episode hash drift")
        features = tf.train.Example.FromString(raw).features.feature
        for timestep in range(int(row["steps"]) - 7):
            for camera in ("image", "wrist_image"):
                encoded = features[f"steps/observation/{camera}"].bytes_list.value[timestep]
                pixels = tf.io.decode_jpeg(encoded, channels=3)
                pixels = tf.image.resize(pixels, [224, 224], method="lanczos3", antialias=True)
                pixels = tf.cast(tf.clip_by_value(tf.round(pixels), 0, 255), tf.uint8).numpy()
                ident = f"{row['episode_id']}_{timestep:04d}_{camera}"
                rel = Path("images") / row["episode_id"] / f"{timestep:04d}_{camera}.png"
                path = args.output / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pixels).save(path)
                entry = {
                    "sample_id": ident,
                    "episode_id": row["episode_id"],
                    "timestep": timestep,
                    "camera": camera,
                    "split": args.split,
                    "task_name": row["task_name"],
                    "instruction": row["language"],
                    "image_path": str(rel),
                    "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "source_jpeg_sha256": hashlib.sha256(encoded).hexdigest(),
                    "source_manifest_sha256": manifest_sha,
                    "preprocessing": "RLDS JPEG->TF Lanczos3 224x224 round uint8; no rotation/crop/augmentation",
                }
                frames.append(entry)
                for role, phrase, occurrence in (
                    ("source", args.source_phrase, "first"),
                    ("target", args.target_phrase, "last"),
                ):
                    semantic.append({
                        **entry,
                        "sample_id": ident + "_" + role,
                        "role": role,
                        "phrase": phrase,
                        "phrase_occurrence": occurrence,
                        "caption": row["language"],
                    })
        print("EPISODE_EXPORTED", row["episode_id"], flush=True)
    for name, data in (("frames", frames), ("semantic", semantic)):
        path = args.output / f"{name}.jsonl"
        path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in data))
        (args.output / f"{name}.sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "\n")
    (args.output / "summary.json").write_text(json.dumps({
        "episodes": len(rows),
        "frames": len(frames),
        "semantic_maps": len(semantic),
        "split": args.split,
        "manifest_sha256": manifest_sha,
        "all_valid_h8_frames": True,
        "camera_count": 2,
        "phrase_mapping": "task-specific source/target; last target mention, no box supervision",
    }, indent=2, ensure_ascii=False) + "\n")
    print("EXPORT_OK", len(frames), len(semantic))


if __name__ == "__main__":
    main()
