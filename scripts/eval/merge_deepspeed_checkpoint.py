#!/usr/bin/env python3
"""Merge a DeepSpeed ZeRO checkpoint into a normal DreamZero eval checkpoint.

This is an offline conversion tool. It does not participate in training.

Example:
python merge_deepspeed_checkpoint.py \
    --zero-checkpoint /meta_eon_cfs/home/lqc/dreamview/checkpoints/robotwin_crossview_full/checkpoint-4000 \
    --output-dir /meta_eon_cfs/home/lqc/dreamview/checkpoints/robotwin_crossview_merged/checkpoint-4000 \
    --dtype bf16 \
    --max-shard-size 5GB

conda activate /meta_eon_cfs/home/lqc/miniconda3/envs/dreamzero

CUDA_VISIBLE_DEVICES=4 python merge_deepspeed_checkpoint.py \
    --zero-checkpoint /meta_eon_cfs/home/lqc/dreamview/checkpoints/robotwin_threeview_baseline_16train_full/checkpoint-35000 \
    --output-dir /meta_eon_cfs/home/lqc/dreamview/checkpoints/robotwin_threeview_baseline_16train_full_merged/checkpoint-35000 \
    --dtype bf16 \
    --max-shard-size 5GB

CUDA_VISIBLE_DEVICES=4 python merge_deepspeed_checkpoint.py \
    --zero-checkpoint /meta_eon_cfs/home/lqc/dreamtriple/checkpoints/robotwin_triple_target_16train_full/checkpoint-35000 \
    --output-dir /meta_eon_cfs/home/lqc/dreamtriple/checkpoints/robotwin_triple_target_16train_full_merged/checkpoint-35000 \
    --dtype bf16 \
    --max-shard-size 5GB

"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import save_file


DTYPE_MAP = {
    # 项目已有 DreamZero 全参和 LoRA checkpoint 都是 BF16 safetensors；
    # 聚合评测权重默认转 BF16，避免导出 FP32 导致磁盘和加载内存翻倍。
    "fp32": torch.float32,
    "float32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
}


def parse_size(size: str) -> int:
    size = size.strip().upper()
    units = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
    }
    for unit, multiplier in sorted(units.items(), key=lambda item: -len(item[0])):
        if size.endswith(unit):
            return int(float(size[: -len(unit)]) * multiplier)
    return int(size)


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def strip_prefix_if_present(key: str, prefixes: Iterable[str]) -> str:
    for prefix in prefixes:
        if prefix and key.startswith(prefix):
            return key[len(prefix) :]
    return key


def normalize_state_dict(
    state_dict: dict[str, torch.Tensor],
    dtype: torch.dtype,
    strip_prefixes: Iterable[str],
) -> OrderedDict[str, torch.Tensor]:
    # DeepSpeed 聚合出来的是训练态 state_dict，可能带 module. 前缀；
    # 这里转换成 DreamZero from_pretrained 期望的普通权重 key。
    normalized = OrderedDict()
    for key in sorted(state_dict.keys()):
        value = state_dict[key]
        if not torch.is_tensor(value):
            continue
        new_key = strip_prefix_if_present(key, strip_prefixes)
        # 聚合结果先落到 CPU，再按导出 dtype 存 safetensors，避免评测 checkpoint 过大。
        tensor = value.detach().cpu().contiguous()
        if tensor.is_floating_point():
            tensor = tensor.to(dtype=dtype)
        normalized[new_key] = tensor
    assert normalized, "No tensor weights found in merged checkpoint."
    return normalized


def save_sharded_safetensors(
    state_dict: OrderedDict[str, torch.Tensor],
    output_dir: Path,
    max_shard_size: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    total_size = sum(tensor_nbytes(tensor) for tensor in state_dict.values())
    shards: list[OrderedDict[str, torch.Tensor]] = []
    current_shard: OrderedDict[str, torch.Tensor] = OrderedDict()
    current_size = 0

    # 按 tensor 顺序贪心切分 shard。DreamZero 的 from_pretrained 支持
    # model.safetensors.index.json，因此这里不需要强行保存成单个大文件。
    for key, tensor in state_dict.items():
        size = tensor_nbytes(tensor)
        if current_shard and current_size + size > max_shard_size:
            shards.append(current_shard)
            current_shard = OrderedDict()
            current_size = 0
        current_shard[key] = tensor
        current_size += size

    if current_shard:
        shards.append(current_shard)

    assert shards, "Internal error: no shards produced."

    weight_map: dict[str, str] = {}
    shard_count = len(shards)
    for shard_index, shard in enumerate(shards, start=1):
        shard_name = f"model-{shard_index:05d}-of-{shard_count:05d}.safetensors"
        shard_path = output_dir / shard_name
        save_file(shard, str(shard_path), metadata={"format": "pt"})
        for key in shard.keys():
            weight_map[key] = shard_name
        print(f"[save] {shard_path} ({len(shard)} tensors)")

    if shard_count == 1:
        # 保持与现有 LoRA checkpoint 一致：单 shard 时直接叫 model.safetensors。
        single_path = output_dir / "model.safetensors"
        first_shard_path = output_dir / "model-00001-of-00001.safetensors"
        first_shard_path.rename(single_path)
        print(f"[save] renamed single shard to {single_path}")
        return

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    index_path = output_dir / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"[save] {index_path}")


def copy_metadata(zero_checkpoint: Path, output_dir: Path) -> None:
    # 权重之外必须保留 config.json，否则 VLA.from_pretrained 无法实例化模型结构。
    required_config = zero_checkpoint / "config.json"
    assert required_config.exists(), (
        f"Missing config.json in {zero_checkpoint}. "
        "The sharded training checkpoint must save model config metadata."
    )
    shutil.copy2(required_config, output_dir / "config.json")

    # 这些文件/目录不是权重本体，但评测脚本可能依赖 tokenizer 或实验元数据。
    for name in ("experiment_cfg", "processor", "tokenizer_config.json", "special_tokens_map.json"):
        source = zero_checkpoint / name
        if not source.exists():
            continue
        target = output_dir / name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)


def resolve_tag(zero_checkpoint: Path, tag: str | None) -> str | None:
    if tag:
        return tag
    # DeepSpeed save_checkpoint 会维护 latest 文件；没有显式 tag 时优先用它。
    latest = zero_checkpoint / "latest"
    if latest.exists():
        value = latest.read_text().strip()
        if value:
            return value
    return None


def merge_checkpoint(args: argparse.Namespace) -> None:
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

    zero_checkpoint = Path(args.zero_checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    assert zero_checkpoint.is_dir(), f"ZeRO checkpoint directory not found: {zero_checkpoint}"
    assert not output_dir.exists() or args.overwrite, (
        f"Output directory already exists: {output_dir}. Pass --overwrite to replace it."
    )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = resolve_tag(zero_checkpoint, args.tag)
    print(f"[load] zero_checkpoint={zero_checkpoint}")
    print(f"[load] tag={tag or '<deepspeed default>'}")
    # 这里会在 CPU 侧从 ZeRO optimizer/model shards 还原 FP32 master weights；
    # 后续再按 --dtype 转成评测权重。不要在训练进程里做这一步。
    state_dict = get_fp32_state_dict_from_zero_checkpoint(str(zero_checkpoint), tag=tag)

    dtype = DTYPE_MAP[args.dtype]
    strip_prefixes = tuple(args.strip_prefix)
    print(f"[convert] tensors={len(state_dict)} dtype={dtype} strip_prefixes={strip_prefixes}")
    normalized = normalize_state_dict(state_dict, dtype=dtype, strip_prefixes=strip_prefixes)
    del state_dict

    save_sharded_safetensors(
        normalized,
        output_dir=output_dir,
        max_shard_size=parse_size(args.max_shard_size),
    )
    copy_metadata(zero_checkpoint, output_dir)
    print(f"[done] merged checkpoint saved to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero-checkpoint", required=True, help="DeepSpeed checkpoint directory.")
    parser.add_argument("--output-dir", required=True, help="Output eval checkpoint directory.")
    parser.add_argument("--tag", default=None, help="DeepSpeed checkpoint tag. Defaults to latest when present.")
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=sorted(DTYPE_MAP.keys()),
        help="Floating dtype used for saved eval weights.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Maximum safetensors shard size, e.g. 2GB, 5GB, 10GB.",
    )
    parser.add_argument(
        "--strip-prefix",
        action="append",
        default=["module."],
        help="State-dict prefix to strip. Can be passed multiple times.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace output directory if it exists.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    merge_checkpoint(args)


if __name__ == "__main__":
    main()
