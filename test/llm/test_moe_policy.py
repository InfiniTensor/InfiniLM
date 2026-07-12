import json

import pytest
from infinilm.llm.llm import (
    _default_max_num_batched_tokens,
    _resolve_moe_backend,
)


def _model_dir(tmp_path, config):
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(tmp_path)


def test_auto_selects_fused_qwen3_moe(tmp_path):
    model_path = _model_dir(tmp_path, {"model_type": "qwen3_moe"})

    assert _resolve_moe_backend(model_path, "cuda", "bfloat16", "auto", None) == (
        True,
        "fused",
    )


@pytest.mark.parametrize(
    ("config", "device", "dtype"),
    [
        ({"model_type": "qwen3_moe"}, "cpu", "bfloat16"),
        ({"model_type": "qwen3_moe"}, "cuda", "float32"),
        ({"model_type": "llama"}, "cuda", "bfloat16"),
        (
            {
                "model_type": "qwen3_moe",
                "quantization_config": {"quant_method": "gptq"},
            },
            "cuda",
            "bfloat16",
        ),
    ],
)
def test_auto_preserves_legacy_for_unsupported_configs(tmp_path, config, device, dtype):
    model_path = _model_dir(tmp_path, config)

    assert _resolve_moe_backend(model_path, device, dtype, "auto", None) == (
        False,
        "legacy",
    )


def test_explicit_backend_and_legacy_override(tmp_path):
    model_path = _model_dir(tmp_path, {"model_type": "llama"})

    assert _resolve_moe_backend(model_path, "cpu", "float32", "fused", None) == (
        True,
        "fused",
    )
    assert _resolve_moe_backend(model_path, "cpu", "float32", "auto", False) == (
        False,
        "legacy",
    )
    with pytest.raises(ValueError, match="Conflicting MoE options"):
        _resolve_moe_backend(model_path, "cpu", "float32", "legacy", True)


def test_qwen3_moe_prefill_budget():
    qwen_config = {"model_type": "qwen3_moe"}

    assert _default_max_num_batched_tokens(qwen_config, 31, 32768) == 32768
    assert _default_max_num_batched_tokens(qwen_config, 32, 32768) == 4096
    assert _default_max_num_batched_tokens(qwen_config, 64, 2048) == 2048
    assert _default_max_num_batched_tokens({"model_type": "llama"}, 64, 32768) == 32768
