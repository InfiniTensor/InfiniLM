#!/usr/bin/env python3
"""Add InfiniLM model metadata to an official Mamba2 checkpoint directory."""

import argparse
import json
from pathlib import Path


def prepare_config(source: dict) -> dict:
    """Translate the state-spaces config into the fields used by InfiniLM."""
    hidden_size = int(source.get("hidden_size", source.get("d_model", 768)))
    intermediate_size = int(
        source.get("intermediate_size", source.get("d_intermediate", 0)) or hidden_size * 2
    )
    # The released state-spaces Mamba2-130M config omits d_state; its
    # projection and convolution shapes identify the trained value as 128.
    state_size = int(source.get("state_size", source.get("d_state", 128)))
    conv_kernel = int(source.get("conv_kernel", source.get("d_conv", 4)))
    num_hidden_layers = int(source.get("num_hidden_layers", source.get("n_layer", 24)))
    head_dim = int(source.get("head_dim", 64))
    num_heads = int(source.get("num_heads", intermediate_size // head_dim))

    vocab_size = int(source.get("vocab_size", 0))
    vocab_multiple = int(source.get("pad_vocab_size_multiple", 1))
    if vocab_multiple > 1:
        vocab_size = (vocab_size + vocab_multiple - 1) // vocab_multiple * vocab_multiple

    config = dict(source)
    config.update(
        {
            "architectures": ["Mamba2ForCausalLM"],
            "model_type": "mamba2",
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": num_hidden_layers,
            "state_size": state_size,
            "conv_kernel": conv_kernel,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "norm_dim": intermediate_size,
            "num_groups": int(source.get("num_groups", 1)),
            "rms_norm_eps": float(source.get("rms_norm_eps", 1e-5)),
            "layer_norm_epsilon": float(source.get("layer_norm_epsilon", 1e-5)),
            "use_bias": bool(source.get("use_bias", False)),
            "use_conv_bias": bool(source.get("use_conv_bias", True)),
            "torch_dtype": source.get("torch_dtype", "float16"),
            "max_position_embeddings": int(
                source.get("max_position_embeddings", source.get("max_seq_len", 2048))
            ),
        }
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=Path)
    args = parser.parse_args()
    config_path = args.checkpoint_dir / "config.json"
    config = json.loads(config_path.read_text())
    config_path.write_text(json.dumps(prepare_config(config), indent=2) + "\n")


if __name__ == "__main__":
    main()
