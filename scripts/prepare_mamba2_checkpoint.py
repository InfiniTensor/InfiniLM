"""Prepare a native state-spaces Mamba-2 checkpoint for the standard loader."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer


def normalize_config(native: dict, tokenizer) -> dict:
    """Make the native Mamba-2 inference defaults explicit."""
    ssm = native.get("ssm_cfg", {})
    if ssm.get("layer") != "Mamba2":
        raise ValueError("Expected a native `Mamba2` checkpoint.")
    if native.get("attn_layer_idx") or native.get("d_intermediate", 0):
        raise ValueError(
            "Hybrid attention and additional MLP layers are not supported."
        )
    hidden = native["d_model"]
    expand = ssm.get("expand", 2)
    state_size = ssm.get("d_state", 128)
    if (
        hidden <= 0
        or expand <= 0
        or native["n_layer"] <= 0
        or not 1 <= state_size <= 256
    ):
        raise ValueError("Model dimensions must be positive and `d_state` at most 256.")
    intermediate = hidden * expand
    if (
        ssm.get("d_ssm", intermediate) not in (None, intermediate)
        or ssm.get("D_has_hdim", False)
        or not ssm.get("rmsnorm", True)
        or ssm.get("norm_before_gate", False)
        or not native.get("rms_norm", True)
        or ssm.get("bias", False)
        or not ssm.get("conv_bias", True)
        or not native.get("residual_in_fp32", True)
        or list(ssm.get("dt_limit", (0.0, float("inf")))) != [0.0, float("inf")]
    ):
        raise ValueError("This checkpoint uses an unsupported Mamba-2 variant.")
    head_dim = ssm.get("headdim", 64)
    groups = ssm.get("ngroups", 1)
    if head_dim <= 0 or groups <= 0 or intermediate % (head_dim * groups):
        raise ValueError("SSM heads must divide evenly into groups.")
    if groups != 1:
        raise ValueError("The current model integration requires `ngroups=1`.")
    if ssm.get("d_conv", 4) != 4:
        raise ValueError("The current causal convolution requires a kernel of size 4.")
    vocab = native["vocab_size"]
    multiple = native.get("pad_vocab_size_multiple", 8)
    if multiple <= 0 or vocab <= 0:
        raise ValueError("Vocabulary size and padding multiple must be positive.")
    padded_vocab = ((vocab + multiple - 1) // multiple) * multiple
    if len(tokenizer) > padded_vocab:
        raise ValueError("The tokenizer vocabulary exceeds the checkpoint vocabulary.")
    return {
        "model_type": "mamba2",
        "architectures": ["Mamba2ForCausalLM"],
        "torch_dtype": "bfloat16",
        "hidden_size": hidden,
        "num_hidden_layers": native["n_layer"],
        "vocab_size": padded_vocab,
        "unpadded_vocab_size": vocab,
        "intermediate_size": intermediate,
        "expand": expand,
        "num_heads": intermediate // head_dim,
        "head_dim": head_dim,
        "n_groups": groups,
        "state_size": state_size,
        "conv_kernel": 4,
        "use_bias": ssm.get("bias", False),
        "use_conv_bias": ssm.get("conv_bias", True),
        "hidden_act": "silu",
        "layer_norm_epsilon": 1e-5,
        "rms_norm_eps": 1e-5,
        "norm_before_gate": False,
        "residual_in_fp32": native.get("residual_in_fp32", True),
        "tie_word_embeddings": native.get("tie_embeddings", True),
        "chunk_size": ssm.get("chunk_size", 256),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": (
            tokenizer.eos_token_id
            if tokenizer.pad_token_id is None
            else tokenizer.pad_token_id
        ),
    }


def prepare_checkpoint(source: Path, output: Path, tokenizer_path: Path) -> None:
    """Convert local files without changing the source or downloading dependencies."""
    if output.exists() and any(output.iterdir()):
        raise ValueError("The output directory must be empty.")
    native = json.loads((source / "config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    config = normalize_config(native, tokenizer)
    weights = torch.load(
        source / "pytorch_model.bin", map_location="cpu", weights_only=True
    )
    expected_embedding = (config["vocab_size"], config["hidden_size"])
    if tuple(weights["backbone.embedding.weight"].shape) != expected_embedding:
        raise ValueError("Embedding shape does not match the normalized configuration.")
    if config["tie_word_embeddings"]:
        head = weights.pop("lm_head.weight", None)
        if head is not None and not torch.equal(
            head, weights["backbone.embedding.weight"]
        ):
            raise ValueError(
                "The checkpoint declares tied but unequal embedding weights."
            )
    # HF and InfiniLM both use `embeddings`; native Mamba uses `embedding`.
    weights["backbone.embeddings.weight"] = weights.pop("backbone.embedding.weight")
    output.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: tensor.contiguous() for name, tensor in weights.items()},
        output / "model.safetensors",
        metadata={"format": "pt"},
    )
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    tokenizer.save_pretrained(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    prepare_checkpoint(args.source, args.output, args.tokenizer)
