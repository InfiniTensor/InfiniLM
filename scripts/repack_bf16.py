#!/usr/bin/env python3
"""Losslessly repack Qwen BF16 projection weights into [K, N] layout."""

import argparse
import json
import os
import re
from pathlib import Path

from safetensors.torch import safe_open, save_file

LINEAR_RE = re.compile(
    r"model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj)\.weight$|"
    r"model\.layers\.\d+\.mlp\.down_proj\.weight$"
)


def convert_shard(source: Path, target: Path, pattern) -> None:
    tensors = {}
    metadata = None
    with safe_open(str(source), framework="pt", device="cpu") as source_file:
        metadata = source_file.metadata()
        for name in source_file.keys():
            tensor = source_file.get_tensor(name)
            tensors[name] = (
                tensor.t().contiguous()
                if pattern.fullmatch(name)
                else tensor.contiguous()
            )
    save_file(tensors, str(target), metadata=metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", choices=["qkv", "down", "mixed"], default="mixed")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    target = args.target.resolve()
    pattern = {
        "qkv": re.compile(
            r"model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj)\.weight$"
        ),
        "down": re.compile(r"model\.layers\.\d+\.mlp\.down_proj\.weight$"),
        "mixed": LINEAR_RE,
    }[args.layout]
    format_name = {
        "qkv": "qkv-v1",
        "down": "down-v1",
        "mixed": "mixed-v1",
    }[args.layout]
    target.mkdir(parents=True, exist_ok=True)

    with (source / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    config["quantization_config"] = {
        "quant_method": "infinilm-bf16-packed",
        "format": format_name,
    }
    with (target / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    with (source / "model.safetensors.index.json").open(encoding="utf-8") as handle:
        index = json.load(handle)
    shard_names = sorted(set(index["weight_map"].values()))
    for shard_name in shard_names:
        convert_shard(source / shard_name, target / shard_name, pattern)
        print(f"converted {shard_name}", flush=True)

    with (target / "model.safetensors.index.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(index, handle, indent=2)
        handle.write("\n")

    for item in source.iterdir():
        if (
            item.name in {"config.json", "model.safetensors.index.json"}
            or item.name in shard_names
        ):
            continue
        if item.is_file():
            link = target / item.name
            if not link.exists():
                os.symlink(item.resolve(), link)
            print(f"linked {item.name}", flush=True)


if __name__ == "__main__":
    main()
