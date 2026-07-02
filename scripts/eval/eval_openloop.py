#!/usr/bin/env python3
"""DreamZero open-loop evaluation.

Loads a DreamZero checkpoint (LoRA or full fine-tune), runs inference on
recorded dataset observations, and compares predicted actions against
ground-truth actions.

Usage (single-GPU):
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/meta_eon_cfs/home/lqc/dreambase \
/meta_eon_cfs/home/szj/miniconda3/envs/dream/bin/python \
  scripts/eval/eval_openloop.py \
  --server-host 127.0.0.1 \
  --server-port 8010 \
  --mode by_chunk
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import av
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from tianshou.data import Batch

# Ensure the project root is on sys.path
_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

DEFAULT_TOKENIZER_PATH = "/meta_eon_cfs/home/lqc/dreambase/checkpoints/umt5-xxl"


_14DOF_DIM_NAMES = [
    "left_x", "left_y", "left_z",
    "left_roll", "left_pitch", "left_yaw",
    "left_gripper",
    "right_x", "right_y", "right_z",
    "right_roll", "right_pitch", "right_yaw",
    "right_gripper",
]
_20DOF_DIM_NAMES = _14DOF_DIM_NAMES + [
    "head_yaw", "head_pitch",
    "lift",
    "chassis_vx", "chassis_vy", "chassis_omega",
]

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def resolve_model_path(model_path: str) -> str:
    """Allow passing either a checkpoint dir or the parent output dir."""
    path = Path(model_path)
    if (path / "experiment_cfg" / "conf.yaml").exists() and (
        (path / "model.safetensors").exists()
        or (path / "adapter_model.safetensors").exists()
        or (path / "global_step500").exists()
    ):
        return str(path)

    checkpoint_dirs = []
    for child in path.glob("checkpoint-*"):
        if child.is_dir() and (child / "experiment_cfg" / "conf.yaml").exists():
            try:
                step = int(child.name.rsplit("-", 1)[1])
            except ValueError:
                step = -1
            checkpoint_dirs.append((step, child))
    if checkpoint_dirs:
        checkpoint_dirs.sort(key=lambda item: item[0])
        return str(checkpoint_dirs[-1][1])

    return str(path)

def load_episode_parquet(dataset_root: str, episode_idx: int) -> pd.DataFrame:
    chunk = episode_idx // 1000
    path = os.path.join(
        dataset_root, "data", f"chunk-{chunk:03d}", f"episode_{episode_idx:06d}.parquet"
    )
    return pd.read_parquet(path)


def load_episode_video_frames(dataset_root: str, episode_idx: int, cam_key: str) -> list:
    chunk = episode_idx // 1000
    path = os.path.join(
        dataset_root, "videos", f"chunk-{chunk:03d}", cam_key, f"episode_{episode_idx:06d}.mp4"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video not found: {path}")
    container = av.open(path)
    frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    container.close()
    return frames


def load_task_text(dataset_root: str) -> str:
    parquet_files = sorted(Path(dataset_root, "data").glob("**/episode_*.parquet"))
    if parquet_files:
        for column in (
            "annotation.task",
            "annotation.language.action_text",
            "annotation.language.language_instruction",
        ):
            try:
                df = pd.read_parquet(parquet_files[0], columns=[column])
                if len(df) > 0:
                    text = str(df[column].iloc[0]).strip()
                    if text:
                        return text
            except Exception:
                pass

    path = os.path.join(dataset_root, "meta", "tasks.jsonl")
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        first = json.loads(f.readline())
    return first.get("task", "")


_MODALITY_14DOF = {
    "state": {
        "left_arm_position":  {"start": 0,  "end": 6},
        "left_gripper":       {"start": 6,  "end": 7},
        "right_arm_position": {"start": 7,  "end": 13},
        "right_gripper":      {"start": 13, "end": 14},
    },
    "action": {
        "left_arm_position":  {"start": 0,  "end": 6},
        "left_gripper":       {"start": 6,  "end": 7},
        "right_arm_position": {"start": 7,  "end": 13},
        "right_gripper":      {"start": 13, "end": 14},
    },
}

_MODALITY_MANIPARENA_EE = {
    "state": {
        "left_ee_pos":       {"start": 0,  "end": 6},
        "left_gripper_pos":  {"start": 6,  "end": 7},
        "right_ee_pos":      {"start": 7,  "end": 13},
        "right_gripper_pos": {"start": 13, "end": 14},
    },
    "action": {
        "left_ee_pos":       {"start": 0,  "end": 6},
        "left_gripper_pos":  {"start": 6,  "end": 7},
        "right_ee_pos":      {"start": 7,  "end": 13},
        "right_gripper_pos": {"start": 13, "end": 14},
    },
}

_MODALITY_20DOF = {
    "state": {
        "left_ee_cartesian_pos":  {"start": 0,  "end": 3},
        "left_ee_rotation":       {"start": 3,  "end": 6},
        "left_gripper":           {"start": 6,  "end": 7},
        "right_ee_cartesian_pos": {"start": 7,  "end": 10},
        "right_ee_rotation":      {"start": 10, "end": 13},
        "right_gripper":          {"start": 13, "end": 14},
        "head_actions":           {"start": 14, "end": 16},
        "height":                 {"start": 16, "end": 17},
        "car_pose":               {"start": 17, "end": 20},
    },
    "action": {
        "left_ee_cartesian_pos":  {"start": 0,  "end": 3},
        "left_ee_rotation":       {"start": 3,  "end": 6},
        "left_gripper":           {"start": 6,  "end": 7},
        "right_ee_cartesian_pos": {"start": 7,  "end": 10},
        "right_ee_rotation":      {"start": 10, "end": 13},
        "right_gripper":          {"start": 13, "end": 14},
        "head_actions":           {"start": 14, "end": 16},
        "height":                 {"start": 16, "end": 17},
        "car_pose":               {"start": 17, "end": 20},
    },
}


def load_modality_from_checkpoint(model_path: str, embodiment_tag: str) -> dict:
    """Read the actual state/action key mapping from the checkpoint's saved config.

    This is the most reliable source because the ConcatTransform inside
    eval_transform uses exactly these keys.
    """
    conf_path = os.path.join(model_path, "experiment_cfg", "conf.yaml")
    if os.path.exists(conf_path):
        import yaml
        with open(conf_path) as f:
            cfg = yaml.safe_load(f)
        mc = cfg.get("modality_configs", {}).get(embodiment_tag, {})
        state_keys = mc.get("state", {}).get("modality_keys", [])
        action_keys = mc.get("action", {}).get("modality_keys", [])
        if state_keys and action_keys:
            return _build_modality_from_keys(state_keys, action_keys)

    if "20dof" in embodiment_tag:
        return _MODALITY_20DOF
    if embodiment_tag == "maniparena_preliminary_ee":
        return _MODALITY_MANIPARENA_EE
    return _MODALITY_14DOF


def _build_modality_from_keys(state_keys: list, action_keys: list) -> dict:
    """Infer start/end offsets from ordered key list + known 14/20-DOF layouts."""
    def _key_dim(key: str) -> int:
        k = key.split(".")[-1]
        if k == "joint_action":
            return 14
        if "gripper" in k:
            return 1
        if k in {"left_ee_pos", "right_ee_pos"}:
            return 6
        if "head" in k:
            return 2
        if "height" in k:
            return 1
        if "car_pose" in k:
            return 3
        if "arm_position" in k:
            return 6
        if "cartesian_pos" in k:
            return 3
        if "rotation" in k:
            return 3
        return 3  # conservative default

    def _make_spec(keys):
        spec = {}
        offset = 0
        for full_key in keys:
            short = full_key.split(".", 1)[-1]  # strip "state." or "action."
            dim = _key_dim(full_key)
            spec[short] = {"start": offset, "end": offset + dim}
            offset += dim
        return spec

    return {"state": _make_spec(state_keys), "action": _make_spec(action_keys)}


def is_maniparena_ee_tag(embodiment_tag: str) -> bool:
    return embodiment_tag == "maniparena_preliminary_ee"


def is_robotwin_tag(embodiment_tag: str) -> bool:
    return embodiment_tag == "robotwin_crossview_joint"


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------

def build_observation(
    face_frame: np.ndarray,
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    state_vector: np.ndarray,
    language: str,
    modality: dict,
    embodiment_tag: str,
    front_frame: np.ndarray | None = None,
) -> dict:
    """Build an observation dict with keys expected by the eval transform pipeline.

    The transform pipeline (VideoToTensor -> VideoResize -> ConcatTransform -> DreamTransform)
    expects per-modality keys: video.faceImg, state.left_ee_cartesian_pos, etc.
    Video values: (T, H, W, C) uint8.  State values: (T, D) float32.
    """
    obs = {}

    def _video_seq(frames: np.ndarray) -> np.ndarray:
        arr = np.asarray(frames)
        if arr.ndim == 3:
            return arr[np.newaxis]
        if arr.ndim != 4:
            raise ValueError(f"Expected video frame or sequence [T,H,W,C], got {arr.shape}")
        return arr

    # --- Video (T=1 at eval) ---
    if is_robotwin_tag(embodiment_tag):
        obs["video.input_head"] = _video_seq(face_frame)
        obs["video.input_left"] = _video_seq(left_frame)
        obs["video.input_right"] = _video_seq(right_frame)
    elif is_maniparena_ee_tag(embodiment_tag):
        obs["video.faceImg"] = _video_seq(face_frame)
        obs["video.leftImg"] = _video_seq(left_frame)
        obs["video.rightImg"] = _video_seq(right_frame)
    else:
        obs["video.face_image"] = _video_seq(face_frame)
        obs["video.left_image"] = _video_seq(left_frame)
        obs["video.right_image"] = _video_seq(right_frame)

    # --- State (split by modality.json) ---
    state_mod = modality.get("state", {})
    for key, spec in state_mod.items():
        s, e = spec["start"], spec["end"]
        obs[f"state.{key}"] = state_vector[np.newaxis, s:e].astype(np.float32)  # (1, D)

    # --- Language ---
    if is_robotwin_tag(embodiment_tag):
        obs["annotation.task"] = language
    elif is_maniparena_ee_tag(embodiment_tag):
        obs["annotation.language.action_text"] = language
    else:
        obs["annotation.language.language_instruction"] = language

    return obs


def reset_policy_causal_state(policy: GrootSimPolicy) -> None:
    action_head = policy.trained_model.action_head
    for name in ("current_start_frame", "skip_countdown"):
        if hasattr(action_head, name):
            setattr(action_head, name, 0)
    for name in ("language", "clip_feas", "ys", "kv_cache1", "kv_cache_neg", "crossattn_cache", "crossattn_cache_neg"):
        if hasattr(action_head, name):
            setattr(action_head, name, None)


def set_persistent_source_cache(policy: GrootSimPolicy, enabled: bool) -> None:
    action_head = policy.trained_model.action_head
    enabled = bool(enabled)
    if hasattr(action_head, "persistent_source_cache"):
        action_head.persistent_source_cache = enabled
    if hasattr(action_head, "config"):
        setattr(action_head.config, "persistent_source_cache", enabled)


def infer_num_frame_per_block(policy: GrootSimPolicy | None, metadata: dict[str, Any] | None) -> int:
    if metadata is not None:
        return int(metadata.get("num_frame_per_block", 2))
    assert policy is not None
    return int(getattr(policy.trained_model.action_head, "num_frame_per_block", 2))


def stack_indices(frames: list[np.ndarray], indices: list[int]) -> np.ndarray:
    return np.stack([frames[int(i)] for i in indices], axis=0)


def action_chunk_to_2d(value: Any, dim: int | None = None) -> np.ndarray:
    """Normalize a model action output to `(horizon, dim)`."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)

    # Single-sample eval often returns `(1, H, D)`; strip singleton batch dims.
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        if dim == 1:
            arr = arr.reshape(-1, 1)
        elif dim is not None and arr.size == dim:
            arr = arr.reshape(1, dim)
        else:
            arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D action chunk, got shape {arr.shape}")

    if dim is not None:
        if arr.shape[1] < dim:
            pad = np.zeros((arr.shape[0], dim - arr.shape[1]), dtype=arr.dtype)
            arr = np.concatenate([arr, pad], axis=-1)
        elif arr.shape[1] > dim:
            arr = arr[:, :dim]

    return arr.astype(np.float32, copy=False)


def infer_action_horizon(act_dict: dict, modality: dict) -> int:
    """Infer the predicted action horizon from the first available action key."""
    action_mod = modality.get("action", {})
    sorted_keys = sorted(action_mod.keys(), key=lambda k: action_mod[k]["start"])
    for key in sorted_keys:
        full_key = f"action.{key}"
        if full_key in act_dict:
            spec = action_mod[key]
            dim = int(spec["end"]) - int(spec["start"])
            return int(action_chunk_to_2d(act_dict[full_key], dim=dim).shape[0])
    return 0


def flatten_action_dict(act_dict: dict, modality: dict) -> np.ndarray:
    """Flatten the model's per-key action output back to a single vector.

    The order follows the modality.json action spec, sorted by 'start' index.
    Only the first timestep (horizon index 0) of each key is used, matching
    the single-step ground-truth action stored in the dataset row.
    """
    action_mod = modality.get("action", {})
    sorted_keys = sorted(action_mod.keys(), key=lambda k: action_mod[k]["start"])

    parts = []
    for key in sorted_keys:
        full_key = f"action.{key}"
        if full_key in act_dict:
            spec = action_mod[key]
            dim = int(spec["end"]) - int(spec["start"])
            chunk = action_chunk_to_2d(act_dict[full_key], dim=dim)
            parts.append(chunk[0].reshape(-1))
    return np.concatenate(parts) if parts else np.array([])


def flatten_action_chunk_full(act_dict: dict, modality: dict) -> np.ndarray:
    """Return the full predicted action chunk as a (horizon, total_dim) array.

    Used by by_chunk mode so one forward pass produces `horizon` predictions
    aligned to t .. t+horizon-1 of the episode.
    """
    action_mod = modality.get("action", {})
    sorted_keys = sorted(action_mod.keys(), key=lambda k: action_mod[k]["start"])

    parts = []
    for key in sorted_keys:
        full_key = f"action.{key}"
        if full_key in act_dict:
            spec = action_mod[key]
            dim = int(spec["end"]) - int(spec["start"])
            parts.append(action_chunk_to_2d(act_dict[full_key], dim=dim))
    return np.concatenate(parts, axis=-1) if parts else np.zeros((0, 0))


def select_action_vector(action_vector: np.ndarray, modality: dict) -> np.ndarray:
    """Select the action dimensions used by the current embodiment.

    Some ManipArena parquet files keep the original 56D robot action, while the
    DreamZero embodiment only trains the first 14D EE subset. The prediction is
    already flattened by modality, so GT must be flattened with the same spec.
    """
    action_mod = modality.get("action", {})
    sorted_items = sorted(action_mod.items(), key=lambda item: item[1]["start"])
    parts = []
    for _key, spec in sorted_items:
        start, end = int(spec["start"]), int(spec["end"])
        parts.append(action_vector[start:end])
    return np.concatenate(parts).astype(np.float32, copy=False) if parts else action_vector


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_openloop(args):
    use_server = bool(args.server_host)
    rank = dist.get_rank() if dist.is_initialized() else 0
    is_main = rank == 0
    args.model_path = resolve_model_path(args.model_path)

    # 1. Load data
    if is_main:
        print(f"[1/4] Loading episode {args.episode} from {args.dataset} ...")
    df = load_episode_parquet(args.dataset, args.episode)
    task_text = load_task_text(args.dataset)
    modality = load_modality_from_checkpoint(args.model_path, args.embodiment_tag)

    if is_robotwin_tag(args.embodiment_tag):
        face_frames = load_episode_video_frames(args.dataset, args.episode, "observation.images.input_head")
        front_frames = None
        left_frames = load_episode_video_frames(args.dataset, args.episode, "observation.images.input_left")
        right_frames = load_episode_video_frames(args.dataset, args.episode, "observation.images.input_right")
        num_frames = min(len(df), len(face_frames), len(left_frames), len(right_frames))
    else:
        face_frames = load_episode_video_frames(args.dataset, args.episode, "observation.images.faceImg")
        front_frames = None
        left_frames = load_episode_video_frames(args.dataset, args.episode, "observation.images.leftImg")
        right_frames = load_episode_video_frames(args.dataset, args.episode, "observation.images.rightImg")
        num_frames = min(len(df), len(face_frames), len(left_frames), len(right_frames))

    if is_main:
        print(f"       Task: {task_text}")
        print(f"       Usable frames: {num_frames}")

    # 2. Load model or connect to persistent server
    policy = None
    client = None
    if use_server:
        from eval_utils.dreambase_server_client import DreamBasePolicyClient

        if is_main:
            print(f"\n[2/4] Connecting to DreamBase server at {args.server_host}:{args.server_port} ...")
        client = DreamBasePolicyClient(args.server_host, args.server_port)
        client.reset({})
        if client.metadata.get("embodiment_tag") != args.embodiment_tag:
            raise ValueError(f"Connected DreamBase server has wrong embodiment: {client.metadata}")
        if client.metadata.get("baseline") != "threeview_original_dreamzero":
            raise ValueError(f"Connected server is not DreamBase three-view baseline: {client.metadata}")
        if args.persistent_source_cache and not bool(client.metadata.get("persistent_source_cache", False)):
            raise ValueError(
                "Open-loop requested --persistent-source-cache, but the connected server "
                "does not enable it. Restart serve_dreambase_policy.py with --persistent-source-cache."
            )
        if is_main:
            print(f"       Connected. Metadata: {client.metadata}")
    else:
        if is_main:
            print(f"\n[2/4] Loading model from {args.model_path} ...")
        embodiment_tag = EmbodimentTag(args.embodiment_tag)
        policy = GrootSimPolicy(
            embodiment_tag=embodiment_tag,
            model_path=args.model_path,
            device=args.device,
            tokenizer_path_override=DEFAULT_TOKENIZER_PATH,
        )
        policy.eval()
        set_persistent_source_cache(policy, args.persistent_source_cache)
        if is_main:
            print("       Model loaded.")

    # 3. Inference loop
    if is_main:
        print(f"\n[3/4] Running open-loop inference ...")

    gt_actions_all = []
    pred_actions_all = []
    inference_times = []
    action_horizon = None
    num_frame_per_block = infer_num_frame_per_block(policy, client.metadata if client is not None else None)
    source_frame_stride = max(1, int(args.source_frame_stride) or (24 // (num_frame_per_block * 4)))
    if is_main:
        print(
            f"       num_frame_per_block={num_frame_per_block}, "
            f"source_frame_stride={source_frame_stride}"
        )

    start_idx = int(num_frames * args.start_ratio)
    end_idx = min(start_idx + args.max_steps, num_frames) if args.max_steps > 0 else num_frames

    # by_chunk mode: stride forward by horizon each outer step, fill in horizon
    # predictions aligned to t..t+H-1. by_frame mode: stride 1, take chunk[0] only.
    by_chunk = args.mode == "by_chunk"
    chunk_boundaries: list[int] = []  # relative offsets from start_idx, for plot

    idx = start_idx
    if by_chunk:
        if use_server:
            assert client is not None
            client.reset({})
        elif policy is not None:
            reset_policy_causal_state(policy)
    while idx < end_idx:
        state_vec = np.array(df["observation.state"].iloc[idx], dtype=np.float32)
        if by_chunk and action_horizon is not None and idx > start_idx:
            source_start = max(start_idx, idx - action_horizon)
            frame_indices = list(range(source_start, idx + 1, source_frame_stride))
            if frame_indices[-1] != idx:
                frame_indices.append(idx)
        else:
            frame_indices = [idx]

        if is_robotwin_tag(args.embodiment_tag):
            face_input = stack_indices(face_frames, frame_indices)
            front_input = None
            left_input = stack_indices(left_frames, frame_indices)
            right_input = stack_indices(right_frames, frame_indices)
        else:
            face_input = face_frames[idx]
            front_input = None
            left_input = left_frames[idx]
            right_input = right_frames[idx]

        obs = build_observation(
            face_frame=face_input,
            left_frame=left_input,
            right_frame=right_input,
            state_vector=state_vec,
            language=task_text,
            modality=modality,
            embodiment_tag=args.embodiment_tag,
            front_frame=front_input,
        )

        try:
            t0 = time.perf_counter()
            if use_server:
                assert client is not None
                response = client.infer(obs)
                act_dict = response["act"]
            else:
                assert policy is not None
                batch = Batch(obs=obs)
                result_batch, _video_pred = policy.lazy_joint_forward_causal(batch)
                act_dict = result_batch.act
            inference_times.append(time.perf_counter() - t0)
            if action_horizon is None:
                action_horizon = infer_action_horizon(act_dict, modality)
                if is_main:
                    debug_shapes = {
                        key: tuple(np.asarray(val).shape)
                        for key, val in act_dict.items()
                    }
                    print(f"       Action output shapes: {debug_shapes}")
                    print(f"       Inferred action horizon per forward: {action_horizon}")
                    print(f"       Mode: {args.mode}  (stride = {action_horizon if by_chunk else 1})")
                    if by_chunk:
                        source_frame_stride = max(
                            1,
                            int(args.source_frame_stride) or (action_horizon // (num_frame_per_block * 4)),
                        )
                        print(f"       KV-cache source frame stride: {source_frame_stride}")
        except Exception as e:
            if is_main:
                print(f"  [ERROR] Step {idx}: {e}")
                import traceback
                traceback.print_exc()
            idx += 1 if not by_chunk else (action_horizon or 1)
            continue

        if by_chunk:
            pred_chunk = flatten_action_chunk_full(act_dict, modality)  # (H, D)
            horizon = pred_chunk.shape[0] if pred_chunk.size else (action_horizon or 1)
            chunk_boundaries.append(idx - start_idx)
            for h in range(horizon):
                t = idx + h
                if t >= end_idx:
                    break
                gt_actions_all.append(select_action_vector(np.array(df["action"].iloc[t], dtype=np.float32), modality))
                pred_actions_all.append(pred_chunk[h])
            idx += horizon
        else:
            gt_actions_all.append(select_action_vector(np.array(df["action"].iloc[idx], dtype=np.float32), modality))
            pred_actions_all.append(flatten_action_dict(act_dict, modality))
            idx += 1

        if is_main and (idx - start_idx) % 50 < (action_horizon or 1):
            print(f"  Step {min(idx, end_idx)}/{end_idx}  source_frames={frame_indices}")

    if not gt_actions_all:
        if is_main:
            print("No successful predictions. Exiting.")
        if use_server:
            assert client is not None
            client.reset({})
        return

    gt_actions_all = np.array(gt_actions_all)
    pred_actions_all = np.array(pred_actions_all)

    action_dim = gt_actions_all.shape[1]
    pred_actions_all = pred_actions_all[:, :action_dim]

    if is_main:
        mse = np.mean((gt_actions_all - pred_actions_all) ** 2)
        print(f"\n  Total steps: {len(gt_actions_all)}")
        print(f"  Overall MSE: {mse:.6f}")
        if inference_times:
            print(f"  Avg wall time / forward: {np.mean(inference_times):.3f}s")

    # 4. Plot
    if is_main:
        print(f"\n[4/4] Plotting ...")
        action_dim = gt_actions_all.shape[1]
        dim_names = (_20DOF_DIM_NAMES if action_dim >= 20 else _14DOF_DIM_NAMES)[:action_dim]
        n_dims = min(action_dim, pred_actions_all.shape[1])

        fig, axes = plt.subplots(n_dims, 1, figsize=(16, 2.8 * n_dims), squeeze=False)
        axes = axes.flatten()
        fig.suptitle(
            f"Open-Loop ({args.mode}): {args.tag}  |  ep={args.episode}  |  task={task_text}",
            fontsize=13, y=1.0,
        )

        for i in range(n_dims):
            ax = axes[i]
            ax.plot(gt_actions_all[:, i], label="Ground Truth", color="blue", alpha=0.8, linewidth=1)
            ax.plot(pred_actions_all[:, i], label="Predicted", color="orange", alpha=0.8, linewidth=1)
            # Mark chunk boundaries (only meaningful in by_chunk mode)
            for bnd in chunk_boundaries[1:]:
                ax.axvline(x=bnd, color="gray", linestyle="--", alpha=0.35, linewidth=0.6)
            name = dim_names[i] if i < len(dim_names) else f"dim_{i}"
            ax.set_ylabel(name, fontsize=9)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time Step")
        plt.tight_layout()

        os.makedirs(args.save_dir, exist_ok=True)
        mode_suffix = "_chunk" if by_chunk else ""
        save_path = os.path.join(
            args.save_dir, f"{args.tag}_ep{args.episode:03d}{mode_suffix}.png"
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved plot → {save_path}")

        npz_path = os.path.join(
            args.save_dir, f"{args.tag}_ep{args.episode:03d}{mode_suffix}.npz"
        )
        np.savez(
            npz_path,
            gt=gt_actions_all,
            pred=pred_actions_all,
            dim_names=dim_names,
            mode=args.mode,
            chunk_boundaries=np.array(chunk_boundaries, dtype=np.int32),
            action_horizon=action_horizon or 0,
            num_frame_per_block=num_frame_per_block,
            source_frame_stride=source_frame_stride,
        )
        print(f"  Saved data → {npz_path}")

    if use_server:
        assert client is not None
        client.reset({})


def main():
    parser = argparse.ArgumentParser(description="DreamZero open-loop evaluation")
    parser.add_argument(
        "--model-path",
        default="/meta_eon_cfs/home/lqc/dreambase/checkpoints/robotwin_threeview_baseline_16train_full/checkpoint-4000",
        help="Path to DreamZero checkpoint or parent output dir",
    )
    parser.add_argument(
        "--dataset",
        default="/meta_eon_cfs/home/lqc/dataset/robotwin_adjust_bottle/dreamzero_robotwin_crossview",
        help="Path to LeRobot dataset",
    )
    parser.add_argument(
        "--embodiment-tag",
        default="robotwin_crossview_joint",
        help="Embodiment tag",
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode index")
    parser.add_argument("--save-dir", default="./eval_results/robotwin_threeview_baseline_openloop", help="Output directory")
    parser.add_argument("--tag", default="dreambase_robotwin_threeview", help="Tag for output filenames")
    parser.add_argument("--device", default="cuda:0", help="Device for inference")
    parser.add_argument("--server-host", default="", help="Use a persistent DreamBase server instead of loading the model locally.")
    parser.add_argument("--server-port", type=int, default=8000, help="Persistent DreamBase server port.")
    parser.add_argument(
        "--persistent-source-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep source KV cache across local-attention windows instead of resetting every 4 chunks.",
    )
    parser.add_argument(
        "--source-frame-stride",
        type=int,
        default=0,
        help="RoboTwin by_chunk KV-cache source stride. 0 means action_horizon / (num_frame_per_block * 4).",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="Max steps (0 = all)")
    parser.add_argument("--start-ratio", type=float, default=0.0, help="Start from this fraction of episode")
    parser.add_argument(
        "--mode",
        choices=["by_frame", "by_chunk"],
        default="by_frame",
        help="by_frame: infer every step, take chunk[0] only (single-step accuracy). "
             "by_chunk: infer every `action_horizon` steps, unfold full chunk (chunk-level open-loop; "
             "reflects real deployment drift).",
    )
    args = parser.parse_args()

    if not args.server_host:
        # Support running with plain `python` (no torchrun needed)
        device_str = str(args.device)
        if device_str.startswith("cuda:"):
            gpu_id = int(device_str.split(":")[1])
            torch.cuda.set_device(gpu_id)

        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29501")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)

    run_openloop(args)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
