#!/usr/bin/env python3
"""Generate deterministic LFM2 reference values without downloading weights.

This script intentionally uses a tiny randomly initialized model.  Its purpose is
to verify model topology and cache semantics before the InfiniLM implementation is
available; it is not an accuracy test for LiquidAI/LFM2-1.2B.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import Lfm2Config, Lfm2ForCausalLM
from transformers.cache_utils import DynamicCache

SEED = 20260907
INPUT_IDS = torch.tensor([[1, 17, 23, 5, 91, 7]], dtype=torch.long)


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().float().cpu()
    flat = value.flatten()
    return {
        "shape": list(value.shape),
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "min": float(value.min()),
        "max": float(value.max()),
        "first_values": [float(x) for x in flat[:8]],
    }


def first_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"No tensor found in hook output of type {type(output)!r}")


def describe_cache(cache: DynamicCache) -> list[dict[str, Any]]:
    description = []
    for layer_idx, layer in enumerate(cache.layers):
        item: dict[str, Any] = {
            "layer_idx": layer_idx,
            "cache_class": type(layer).__name__,
        }
        if hasattr(layer, "conv_states"):
            state = layer.conv_states[0]
            item["conv_state_shape"] = list(state.shape)
            item["has_previous_state"] = bool(layer.has_previous_state[0])
        else:
            item["key_shape"] = list(layer.keys.shape)
            item["value_shape"] = list(layer.values.shape)
        description.append(item)
    return description


def verify_short_conv(
    model: Lfm2ForCausalLM,
    input_ids: torch.Tensor,
) -> float:
    """Rebuild layer-0 ShortConv from primitive PyTorch operations."""
    layer = model.model.layers[0]
    hidden = model.model.embed_tokens(input_ids)
    normalized = layer.operator_norm(hidden)

    projected = layer.conv.in_proj(normalized).transpose(-1, -2)
    b_gate, c_gate, x_value = projected.chunk(3, dim=-2)
    bx = b_gate * x_value

    conv_out = F.conv1d(
        bx,
        layer.conv.conv.weight,
        layer.conv.conv.bias,
        padding=layer.conv.L_cache - 1,
        groups=model.config.hidden_size,
    )[..., : input_ids.shape[1]]
    manual = layer.conv.out_proj((c_gate * conv_out).transpose(-1, -2).contiguous())
    reference = layer.conv(normalized, attention_mask=torch.ones_like(input_ids))
    return float((manual - reference).detach().abs().max())


def run(config_path: Path) -> dict[str, Any]:
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)

    config = Lfm2Config.from_json_file(str(config_path))
    model = Lfm2ForCausalLM(config).float().eval()
    first_attention_idx = config.layer_types.index("full_attention")

    captured: dict[str, dict[str, Any]] = {}

    def capture(name: str):
        def hook(_module, _inputs, output):
            captured[name] = tensor_summary(first_tensor(output))

        return hook

    handles = [
        model.model.layers[0].conv.register_forward_hook(capture("layer0_short_conv")),
        model.model.layers[first_attention_idx].self_attn.register_forward_hook(
            capture(f"layer{first_attention_idx}_attention")
        ),
        model.model.layers[0].feed_forward.register_forward_hook(capture("layer0_mlp")),
        model.model.embedding_norm.register_forward_hook(capture("final_rms_norm")),
    ]

    attention_mask = torch.ones_like(INPUT_IDS)
    with torch.inference_mode():
        full_logits = model(
            input_ids=INPUT_IDS,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits

    for handle in handles:
        handle.remove()

    with torch.inference_mode():
        cache = DynamicCache(config=config)
        prefill_logits = model(
            input_ids=INPUT_IDS[:, :-1],
            attention_mask=attention_mask[:, :-1],
            past_key_values=cache,
            use_cache=True,
        ).logits
        decode_logits = model(
            input_ids=INPUT_IDS[:, -1:],
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        ).logits

        cache_description = describe_cache(cache)
        cache.reset()
        reset_flags = [
            bool(layer.has_previous_state[0])
            for layer in cache.layers
            if hasattr(layer, "has_previous_state")
        ]

    cached_error = float((full_logits[:, -1] - decode_logits[:, -1]).abs().max())
    short_conv_error = verify_short_conv(model, INPUT_IDS)

    torch.testing.assert_close(
        decode_logits[:, -1],
        full_logits[:, -1],
        rtol=1e-5,
        atol=1e-6,
    )
    if short_conv_error > 1e-7:
        raise AssertionError(f"ShortConv reconstruction error is {short_conv_error}")
    if any(reset_flags):
        raise AssertionError("cache.reset() left a convolution state active")

    return {
        "purpose": "LFM2 topology and cache reference; not pretrained-model accuracy",
        "seed": SEED,
        "input_ids": INPUT_IDS.tolist(),
        "config": {
            "hidden_size": config.hidden_size,
            "intermediate_size_before_auto_adjust": config.intermediate_size,
            "actual_mlp_intermediate_size": model.model.layers[
                0
            ].feed_forward.w1.out_features,
            "num_hidden_layers": config.num_hidden_layers,
            "layer_types": config.layer_types,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": model.model.layers[first_attention_idx].self_attn.head_dim,
            "conv_L_cache": config.conv_L_cache,
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "state_dict_shapes": {
            name: list(tensor.shape) for name, tensor in model.state_dict().items()
        },
        "forward": {
            "full_logits": tensor_summary(full_logits),
            "prefill_logits_shape": list(prefill_logits.shape),
            "decode_logits_shape": list(decode_logits.shape),
            "full_last_token_argmax": int(full_logits[:, -1].argmax(dim=-1).item()),
            "cached_last_token_argmax": int(decode_logits[:, -1].argmax(dim=-1).item()),
            "full_vs_cached_max_abs_error": cached_error,
            "short_conv_manual_max_abs_error": short_conv_error,
        },
        "captured_modules": captured,
        "cache_after_prefill_and_decode": cache_description,
        "conv_cache_flags_after_reset": reset_flags,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("tiny_config.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. The result is always printed as well.",
    )
    args = parser.parse_args()

    result = run(args.config)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
