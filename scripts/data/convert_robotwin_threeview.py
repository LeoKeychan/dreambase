from __future__ import annotations

"""Convert RoboTwin HDF5 demos to DreamBase three-view LeRobot format.

DreamBase baseline uses only observed RGB views:
  video.input_head, video.input_left, video.input_right

No target_front video is read or written here.  The embodiment tag intentionally
stays robotwin_crossview_joint because the DreamBase config uses that tag for
the RoboTwin 14D joint-action baseline.
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


SOURCE_CAMERA_KEYS = {
    "input_head": "observation/head_camera/rgb",
    "input_left": "observation/left_camera/rgb",
    "input_right": "observation/right_camera/rgb",
}
VIDEO_FEATURE_PREFIX = "observation.images."
TASK_KEY = "annotation.task"
EMBODIMENT_TAG = "robotwin_crossview_joint"


def decode_hdf5_rgb(raw) -> np.ndarray:
    if isinstance(raw, np.ndarray):
        raw = raw.tobytes()
    if isinstance(raw, np.bytes_):
        raw = bytes(raw)
    encoded = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode HDF5 RGB frame bytes.")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def resize_rgb(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def write_rgb_video(frames: list[np.ndarray], output_path: Path, fps: int, size: tuple[int, int]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")
    for frame in frames:
        frame = resize_rgb(frame, size)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def load_instruction(instruction_path: Path) -> str:
    if not instruction_path.exists():
        return "Pick up the bottle and keep it upright."
    with open(instruction_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ("seen", "unseen"):
        values = data.get(key)
        if isinstance(values, list) and values:
            return str(values[0])
    return "Pick up the bottle and keep it upright."


def statistical_values(values: np.ndarray) -> dict:
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_episode_ids(value: str) -> list[int]:
    if "-" in value:
        start, end = [int(x) for x in value.split("-", 1)]
        return list(range(start, end + 1))
    return [int(x) for x in value.split(",") if x.strip()]


def convert_episode(
    episode_id: int,
    demo_root: Path,
    output_root: Path,
    fps: int,
    source_size: tuple[int, int],
    task_index: int,
    global_index_start: int,
) -> tuple[dict, pd.DataFrame, dict]:
    hdf5_path = demo_root / "data" / f"episode{episode_id}.hdf5"
    if not hdf5_path.exists():
        raise FileNotFoundError(hdf5_path)

    with h5py.File(hdf5_path, "r") as h5:
        actions = np.asarray(h5["joint_action/vector"], dtype=np.float32)
        num_frames = actions.shape[0]
        source_frames_by_key = {
            out_key: [decode_hdf5_rgb(raw) for raw in h5[h5_key][:]]
            for out_key, h5_key in SOURCE_CAMERA_KEYS.items()
        }
        for out_key, frames in source_frames_by_key.items():
            if len(frames) != num_frames:
                raise ValueError(f"episode{episode_id} {out_key} has {len(frames)} frames, expected {num_frames}")

    chunk_dir = output_root / "videos" / "chunk-000"
    for out_key, frames in source_frames_by_key.items():
        write_rgb_video(
            frames,
            chunk_dir / f"{VIDEO_FEATURE_PREFIX}{out_key}" / f"episode_{episode_id:06d}.mp4",
            fps,
            source_size,
        )

    instruction = load_instruction(demo_root / "instructions" / f"episode{episode_id}.json")
    timestamps = np.arange(num_frames, dtype=np.float32) / float(fps)
    rows = []
    for frame_index in range(num_frames):
        rows.append(
            {
                "observation.state": actions[frame_index].astype(np.float32),
                "action": actions[frame_index].astype(np.float32),
                "timestamp": float(timestamps[frame_index]),
                "frame_index": frame_index,
                "episode_index": episode_id,
                "index": global_index_start + frame_index,
                "task_index": task_index,
                TASK_KEY: instruction,
            }
        )
    df = pd.DataFrame(rows)

    parquet_path = output_root / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)

    episode_meta = {
        "episode_index": episode_id,
        "task_index": task_index,
        "tasks": [instruction],
        "length": num_frames,
    }
    return episode_meta, df, {"task_index": task_index, "task": instruction}


def build_info(
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    fps: int,
    source_size: tuple[int, int],
) -> dict:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [14],
            "names": [f"state_{i}" for i in range(14)],
        },
        "action": {
            "dtype": "float32",
            "shape": [14],
            "names": [f"action_{i}" for i in range(14)],
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        TASK_KEY: {"dtype": "string", "shape": [1], "names": None},
    }
    width, height = source_size
    for key in SOURCE_CAMERA_KEYS:
        feature_key = f"{VIDEO_FEATURE_PREFIX}{key}"
        features[feature_key] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": fps,
                "video.height": height,
                "video.width": width,
                "video.channels": 3,
            },
        }
    return {
        "codebase_version": "v2.0",
        "robot_type": EMBODIMENT_TAG,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes * len(SOURCE_CAMERA_KEYS),
        "chunks_size": 1000,
        "fps": fps,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def build_modality() -> dict:
    return {
        "state": {
            "joint_action": {
                "original_key": "observation.state",
                "start": 0,
                "end": 14,
                "rotation_type": None,
                "absolute": True,
                "dtype": "float32",
                "range": None,
            }
        },
        "action": {
            "joint_action": {
                "original_key": "action",
                "start": 0,
                "end": 14,
                "rotation_type": None,
                "absolute": True,
                "dtype": "float32",
                "range": None,
            }
        },
        "video": {
            key: {"original_key": f"{VIDEO_FEATURE_PREFIX}{key}"}
            for key in SOURCE_CAMERA_KEYS
        },
        "annotation": {
            "task": {"original_key": TASK_KEY},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-root", type=Path, default=Path("/meta_eon_cfs/home/lqc/dataset/robotwin/clean_50/adjust_bottle/aloha-agilex_clean_50"))
    parser.add_argument("--output-root", type=Path, default=Path("/meta_eon_cfs/home/lqc/dataset/robotwin/dreamzero_robotwin_threeview_clean_50/adjust_bottle"))
    parser.add_argument("--episodes", type=str, default="0-49", help="Episode ids, e.g. 0-49 or 0,1,2")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--source-width", type=int, default=320)
    parser.add_argument("--source-height", type=int, default=176)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_size = (args.source_width, args.source_height)
    if source_size != (320, 176):
        raise ValueError(
            "DreamBase RoboTwin source views are expected to be 320x176 each. "
            f"Got {args.source_width}x{args.source_height}."
        )

    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists. Pass --overwrite to replace it.")
        shutil.rmtree(args.output_root)
    (args.output_root / "meta").mkdir(parents=True, exist_ok=True)

    episode_ids = parse_episode_ids(args.episodes)
    episodes = []
    dfs = []
    task_rows_by_text = {}
    global_index = 0
    for episode_id in tqdm(episode_ids, desc="convert RoboTwin three-view episodes"):
        instruction = load_instruction(args.demo_root / "instructions" / f"episode{episode_id}.json")
        if instruction not in task_rows_by_text:
            task_rows_by_text[instruction] = {
                "task_index": len(task_rows_by_text),
                "task": instruction,
            }
        task_index = task_rows_by_text[instruction]["task_index"]
        episode_meta, df, task_row = convert_episode(
            episode_id,
            args.demo_root,
            args.output_root,
            args.fps,
            source_size,
            task_index,
            global_index,
        )
        global_index += len(df)
        episodes.append(episode_meta)
        dfs.append(df)
        task_rows_by_text.setdefault(task_row["task"], task_row)

    all_df = pd.concat(dfs, ignore_index=True)
    stats = {
        "observation.state": statistical_values(np.stack(all_df["observation.state"].values)),
        "action": statistical_values(np.stack(all_df["action"].values)),
    }

    meta_dir = args.output_root / "meta"
    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(build_info(len(episode_ids), len(all_df), len(task_rows_by_text), args.fps, source_size), f, indent=2)
    with open(meta_dir / "modality.json", "w", encoding="utf-8") as f:
        json.dump(build_modality(), f, indent=2)
    with open(meta_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    with open(meta_dir / "embodiment.json", "w", encoding="utf-8") as f:
        json.dump({"embodiment_tag": EMBODIMENT_TAG}, f, indent=2)

    write_jsonl(meta_dir / "episodes.jsonl", episodes)
    write_jsonl(
        meta_dir / "tasks.jsonl",
        list(sorted(task_rows_by_text.values(), key=lambda x: x["task_index"]))
        or [{"task_index": 0, "task": "Pick up the bottle and keep it upright."}],
    )

    print(f"Wrote DreamBase RoboTwin three-view dataset to {args.output_root}")
    print(f"Episodes: {len(episode_ids)}, frames: {len(all_df)}, source size: {args.source_width}x{args.source_height}")


if __name__ == "__main__":
    main()
