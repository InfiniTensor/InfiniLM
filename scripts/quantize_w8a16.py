#!/usr/bin/env python3
"""Convert Qwen dense projection weights to InfiniLM W8A16 format."""

import argparse
import json
import os
import re
from pathlib import Path

import torch
from safetensors.torch import safe_open, save_file

LINEAR_RE = re.compile(
    r"model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)\.weight$|"
    r"model\.layers\.\d+\.mlp\.(?:gate_proj|up_proj|down_proj)\.weight$"
)


def quantize_weight(weight: torch.Tensor):
    scale = weight.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
    scale = scale.to(weight.dtype).view(1, weight.shape[0])
    quantized = torch.round(weight.float() / scale.float().view(-1, 1))
    quantized = quantized.clamp(-128, 127).to(torch.int8)
    return quantized.contiguous(), scale.contiguous()


def convert_shard(src: Path, dst: Path):
    tensors = {}
    metadata = None
    with safe_open(str(src), framework="pt", device="cpu") as source:
        metadata = source.metadata()
        for name in source.keys():
            tensor = source.get_tensor(name)
            if LINEAR_RE.fullmatch(name):
                quantized, scale = quantize_weight(tensor)
                tensors[name] = quantized
                tensors[name.replace(".weight", ".weight_scale")] = scale
            else:
                tensors[name] = tensor.contiguous()
    save_file(tensors, str(dst), metadata=metadata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    target = args.target.resolve()
    target.mkdir(parents=True, exist_ok=True)

    with (source / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    config["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "infinilm-w8a16",
        "config_format": "compressed-tensors",
    }
    with (target / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    with (source / "model.safetensors.index.json").open(encoding="utf-8") as handle:
        index = json.load(handle)
    shard_names = sorted(set(index["weight_map"].values()))
    for shard_name in shard_names:
        convert_shard(source / shard_name, target / shard_name)
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
