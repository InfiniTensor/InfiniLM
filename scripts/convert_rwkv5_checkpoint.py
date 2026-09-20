#!/usr/bin/env python3
"""Convert an official RWKV-5 .pth checkpoint to InfiniLM safetensors."""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


def _load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("RWKV checkpoint must contain a state dictionary")
    return {key: value.contiguous() for key, value in checkpoint.items()}


def _infer_config(state_dict: dict[str, torch.Tensor], context_length: int) -> dict:
    if any("time_maa" in key for key in state_dict):
        raise ValueError("This converter supports RWKV-5 only, not RWKV-6 or newer")
    if "blocks.0.att.time_decay" not in state_dict:
        raise ValueError("Checkpoint does not contain RWKV-5 time_decay weights")

    embedding = state_dict["emb.weight"]
    vocab_size, hidden_size = embedding.shape
    layer_ids = {
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("blocks.")
    }
    num_layers = max(layer_ids) + 1
    intermediate_size = state_dict["blocks.0.ffn.key.weight"].shape[0]
    time_decay = state_dict["blocks.0.att.time_decay"].squeeze()
    if time_decay.ndim == 1:
        # RWKV-5.0 stores one decay value per head. The value is shared by all
        # channels in that head, so the converter expands it during loading.
        num_heads = time_decay.numel()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} is not divisible by heads={num_heads}"
            )
        head_dim = hidden_size // num_heads
        rwkv_version = "5.0"
    elif time_decay.ndim == 2:
        num_heads, head_dim = time_decay.shape
        rwkv_version = "5.2"
    else:
        raise ValueError(
            "RWKV-5 time_decay must have shape [num_heads] or "
            "[num_heads, head_dim]"
        )
    if hidden_size != num_heads * head_dim:
        raise ValueError(
            f"hidden_size={hidden_size} does not match heads={num_heads} x head_dim={head_dim}"
        )

    use_gate = "blocks.0.att.gate.weight" in state_dict
    if rwkv_version == "5.0" and use_gate:
        rwkv_version = "5.1"
    if use_gate and "blocks.0.att.time_mix_g" not in state_dict:
        raise ValueError("Gated RWKV-5 checkpoint is missing time_mix_g")
    if not any(
        key in state_dict
        for key in ("blocks.0.att.time_first", "blocks.0.att.time_faaaa")
    ):
        raise ValueError("RWKV-5 checkpoint is missing time_first/time_faaaa")
    if any(key.endswith(".time_state") for key in state_dict):
        raise ValueError("Checkpoints with learned time_state are not supported yet")

    dtype_name = str(embedding.dtype).removeprefix("torch.")
    return {
        "architectures": ["RWKV5ForCausalLM"],
        "model_type": "rwkv5",
        "rwkv_version": rwkv_version,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "head_dim": head_dim,
        "context_length": context_length,
        "max_position_embeddings": context_length,
        "layer_norm_eps": 1e-5,
        "group_norm_eps": 64e-5,
        "use_attention_gate": use_gate,
        "tie_word_embeddings": False,
        "bos_token_id": 0,
        "eos_token_id": 0,
        "pad_token_id": 0,
        "torch_dtype": dtype_name,
    }


def convert(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dict = _load_checkpoint(checkpoint_path)
    config = _infer_config(state_dict, args.context_length)
    save_file(
        state_dict,
        output_dir / "model.safetensors",
        metadata={"format": "pt", "source": checkpoint_path.name},
    )
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "generation_config.json").write_text(
        json.dumps(
            {"bos_token_id": 0, "eos_token_id": 0, "pad_token_id": 0},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    vocab_path = Path(args.vocab).expanduser().resolve()
    if not vocab_path.exists():
        raise FileNotFoundError(f"RWKV vocabulary does not exist: {vocab_path}")
    shutil.copy2(vocab_path, output_dir / "rwkv_vocab_v20230424.txt")
    print(f"Converted {checkpoint_path.name} -> {output_dir}")
    print(json.dumps(config, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Official RWKV-5 .pth checkpoint")
    parser.add_argument("output_dir", help="Destination model directory")
    parser.add_argument(
        "--vocab",
        required=True,
        help="Path to rwkv_vocab_v20230424.txt",
    )
    parser.add_argument("--context-length", type=int, default=4096)
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
