#!/usr/bin/env python3
"""Serve a DreamBase/DreamZero baseline checkpoint over a raw-observation websocket API.

This server is for the three-view baseline path: observations contain
`video.input_head/left/right`, and the model predicts the same three-view
video/action objective. It intentionally does not use DreamView's
source-conditioned `video.target_front` path.

CUDA_VISIBLE_DEVICES=6 \
PYTHONPATH=/meta_eon_cfs/home/lqc/dreambase \
/meta_eon_cfs/home/szj/miniconda3/envs/dream/bin/python \
  eval_utils/serve_dreambase_policy.py \
  --model-path checkpoints/robotwin_dreambase_merged/checkpoint-12000 \
  --device cuda:0 \
  --port 8010 \
  --save-video-dir video_pred/ckpt12000 \
  --save-video-fps 10 


# 维护持久 cache
CUDA_VISIBLE_DEVICES=6 \
PYTHONPATH=/meta_eon_cfs/home/lqc/dreambase \
/meta_eon_cfs/home/szj/miniconda3/envs/dream/bin/python \
  eval_utils/serve_dreambase_policy.py \
  --model-path checkpoints/robotwin_threeview_baseline_16train_full_merged/checkpoint-35000 \
  --device cuda:0 \
  --port 8010 \
  --master-port 29653 \
  --save-video-dir video_pred/ckpt35000 \
  --save-video-fps 10 \
  --persistent-source-cache

"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import numpy as np
import torch
import torch.distributed as dist
import websockets.asyncio.server
import websockets.frames
import imageio.v3 as iio
from openpi_client import msgpack_numpy
from tianshou.data import Batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy


logger = logging.getLogger("serve_dreambase_policy")

DEFAULT_TOKENIZER_PATH = "/meta_eon_cfs/home/lqc/dreambase/checkpoints/umt5-xxl"
DEFAULT_EMBODIMENT_TAG = "robotwin_crossview_joint"


def init_dist(device: str, master_port: str) -> None:
    if device.startswith("cuda:"):
        torch.cuda.set_device(int(device.split(":", 1)[1]))
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", master_port)
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


def resolve_model_path(model_path: str) -> str:
    path = Path(model_path).expanduser()
    if (path / "experiment_cfg" / "conf.yaml").exists():
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


def action_batch_to_dict(action_batch: Any) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if hasattr(action_batch, "items"):
        iterator = action_batch.items()
    else:
        iterator = (
            (name, getattr(action_batch, name))
            for name in dir(action_batch)
            if name.startswith("action.")
        )
    for key, value in iterator:
        if not str(key).startswith("action."):
            continue
        if torch.is_tensor(value):
            value = value.detach().float().cpu().numpy()
        else:
            value = np.asarray(value)
        out[str(key)] = value
    return out


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


def decode_video_pred(policy: GrootSimPolicy, video_pred: torch.Tensor | None) -> np.ndarray | None:
    if video_pred is None:
        return None
    action_head = policy.trained_model.action_head
    latents = video_pred.to(device=policy.device, dtype=torch.bfloat16)
    decoded = action_head.vae.decode(
        latents,
        tiled=action_head.tiled,
        tile_size=(action_head.tile_size_height, action_head.tile_size_width),
        tile_stride=(action_head.tile_stride_height, action_head.tile_stride_width),
    )
    if decoded.ndim == 5:
        decoded = decoded[0]
    decoded = decoded.detach().float().cpu().clamp(-1, 1)
    decoded = ((decoded + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return decoded.permute(1, 2, 3, 0).numpy()


def save_rgb_video(path: Path, frames: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames, dtype=np.uint8)
    iio.imwrite(path, frames, fps=fps)


def stitch_decoded_chunks(chunks: list[np.ndarray], *, drop_first_frame: bool) -> np.ndarray | None:
    frames: list[np.ndarray] = []
    for chunk in chunks:
        if chunk is None or len(chunk) == 0:
            continue
        current = chunk[1:] if drop_first_frame and len(chunk) > 1 else chunk
        frames.extend(list(current))
    if not frames:
        return None
    return np.stack(frames, axis=0)


class DreamBaseRawPolicyServer:
    def __init__(
        self,
        policy: GrootSimPolicy,
        *,
        host: str,
        port: int,
        metadata: dict[str, Any],
        save_video_dir: Path | None = None,
        save_video_fps: int = 6,
        save_video_every: int = 1,
        save_video_max: int = 0,
        save_video_chunks: bool = False,
        save_video_drop_first_frame: bool = True,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata
        self._save_video_dir = save_video_dir
        self._save_video_fps = save_video_fps
        self._save_video_every = max(1, save_video_every)
        self._save_video_max = max(0, save_video_max)
        self._save_video_chunks = save_video_chunks
        self._save_video_drop_first_frame = save_video_drop_first_frame
        self._episode_index = 0
        self._call_index = 0
        self._saved_videos = 0
        self._episode_video_chunks: list[np.ndarray] = []

    def _reset_session(self, *, advance_episode: bool) -> None:
        if advance_episode:
            self._flush_episode_video()
        reset_policy_causal_state(self._policy)
        self._call_index = 0
        if advance_episode:
            self._episode_index += 1

    def _should_save_video(self) -> bool:
        if self._save_video_dir is None:
            return False
        if self._save_video_max > 0 and self._saved_videos >= self._save_video_max:
            return False
        return self._call_index % self._save_video_every == 0

    def _record_video_if_enabled(self, video_pred: torch.Tensor | None) -> Path | None:
        if not self._should_save_video():
            return None
        decoded = decode_video_pred(self._policy, video_pred)
        if decoded is None:
            return None
        assert self._save_video_dir is not None
        self._episode_video_chunks.append(decoded)
        self._saved_videos += 1
        if not self._save_video_chunks:
            return None
        chunk_dir = self._save_video_dir / "chunks"
        path = chunk_dir / f"episode_{self._episode_index:04d}_call_{self._call_index:06d}.mp4"
        save_rgb_video(path, decoded, fps=self._save_video_fps)
        logger.info("Saved predicted chunk video to %s", path)
        return path

    def _flush_episode_video(self) -> Path | None:
        if self._save_video_dir is None or not self._episode_video_chunks:
            self._episode_video_chunks.clear()
            return None
        stitched = stitch_decoded_chunks(
            self._episode_video_chunks,
            drop_first_frame=self._save_video_drop_first_frame,
        )
        self._episode_video_chunks.clear()
        if stitched is None:
            return None
        path = self._save_video_dir / f"episode_{self._episode_index:04d}_stitched.mp4"
        save_rgb_video(path, stitched, fps=self._save_video_fps)
        logger.info("Saved stitched episode video to %s (%d frames)", path, len(stitched))
        return path

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            ping_interval=60,
            ping_timeout=1200,
        ) as server:
            logger.info("DreamBase server listening on %s:%d", self._host, self._port)
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        self._reset_session(advance_episode=False)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))
        while True:
            try:
                request = msgpack_numpy.unpackb(await websocket.recv())
                endpoint = request.pop("endpoint", "infer")
                if endpoint == "reset":
                    self._reset_session(advance_episode=True)
                    await websocket.send(packer.pack({"ok": True}))
                    continue
                if endpoint != "infer":
                    raise ValueError(f"Unknown endpoint: {endpoint}")

                return_video_pred = bool(request.pop("_return_video_pred", False))
                return_decoded_video = bool(request.pop("_return_decoded_video", False))
                request.pop("_client_send_time", None)

                start = time.perf_counter()
                batch = Batch(obs=request)
                with torch.inference_mode():
                    result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
                action_dict = action_batch_to_dict(result_batch.act)
                response: dict[str, Any] = {
                    "act": action_dict,
                    "server_inference_time": np.asarray(time.perf_counter() - start, dtype=np.float32),
                }
                if return_video_pred and video_pred is not None:
                    response["video_pred"] = video_pred.detach().float().cpu().numpy()
                saved_video_path = self._record_video_if_enabled(video_pred)
                if saved_video_path is not None:
                    response["saved_video_path"] = str(saved_video_path)
                if return_decoded_video:
                    decoded = decode_video_pred(self._policy, video_pred)
                    if decoded is not None:
                        response["decoded_video"] = decoded
                await websocket.send(packer.pack(response))
                self._call_index += 1
            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                self._reset_session(advance_episode=True)
                break
            except Exception:
                err = traceback.format_exc()
                logger.error("Server error:\n%s", err)
                self._reset_session(advance_episode=True)
                await websocket.send(err)
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve a DreamBase baseline checkpoint for open-loop/eval scripts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--embodiment-tag", default=DEFAULT_EMBODIMENT_TAG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--master-port", default="29642")
    parser.add_argument("--save-video-dir", default="", help="Server-side directory for decoded predicted mp4 videos. Empty disables saving.")
    parser.add_argument("--save-video-fps", type=int, default=6)
    parser.add_argument("--save-video-every", type=int, default=1, help="Save every N inference calls when --save-video-dir is set.")
    parser.add_argument("--save-video-max", type=int, default=0, help="Maximum server-side videos to save; 0 means unlimited.")
    parser.add_argument("--save-video-chunks", action="store_true", help="Also save per-inference chunk mp4 files under chunks/.")
    parser.add_argument(
        "--persistent-source-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep DreamBase source KV cache across local-attention windows instead of resetting every 4 chunks.",
    )
    parser.add_argument(
        "--save-video-drop-first-frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop the first decoded frame of each chunk when stitching an episode video.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    init_dist(args.device, args.master_port)
    if not Path(DEFAULT_TOKENIZER_PATH).exists():
        raise FileNotFoundError(f"DreamZero tokenizer not found: {DEFAULT_TOKENIZER_PATH}")

    model_path = resolve_model_path(args.model_path)
    logger.info("Loading policy from %s", model_path)
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
        model_path=model_path,
        device=args.device,
        tokenizer_path_override=DEFAULT_TOKENIZER_PATH,
    )
    policy.eval()
    action_head = policy.trained_model.action_head
    set_persistent_source_cache(policy, args.persistent_source_cache)
    metadata = {
        "server": "dreambase_raw_policy",
        "model_path": model_path,
        "embodiment_tag": args.embodiment_tag,
        "baseline": "threeview_original_dreamzero",
        "num_frame_per_block": int(getattr(action_head, "num_frame_per_block", 1)),
        "num_action_per_block": int(getattr(action_head.model, "num_action_per_block", 24)),
        "local_attn_size": int(getattr(action_head.model, "local_attn_size", -1)),
        "persistent_source_cache": bool(args.persistent_source_cache),
        "save_video_dir": args.save_video_dir,
    }
    logger.info("Policy loaded; metadata=%s", metadata)
    DreamBaseRawPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
        save_video_dir=Path(args.save_video_dir).expanduser().resolve() if args.save_video_dir else None,
        save_video_fps=args.save_video_fps,
        save_video_every=args.save_video_every,
        save_video_max=args.save_video_max,
        save_video_chunks=args.save_video_chunks,
        save_video_drop_first_frame=args.save_video_drop_first_frame,
    ).serve_forever()


if __name__ == "__main__":
    main()
