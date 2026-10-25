"""Model dtype overrides reach the native parameter and cache configuration."""

import json

import pytest
from infinilm import _core as infinicore
from infinilm.cache.cache import StaticKVCacheConfig
from infinilm.config.engine_config import EngineConfig
from infinilm.infer_engine import InferEngine
from infinilm.llm.model_runner.model_runner import ModelRunner


@pytest.fixture
def checkpoint(tmp_path):
    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "torch_dtype": "float32",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 32,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "hidden_act": "silu",
        "attention_bias": False,
        "mlp_bias": False,
        "tie_word_embeddings": False,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


@pytest.mark.parametrize(
    "override,expected",
    [
        (None, "float32"),
        ("float16", "float16"),
        ("bfloat16", "bfloat16"),
    ],
)
def test_infer_engine_native_parameter_dtype(checkpoint, override, expected):
    engine = InferEngine(
        checkpoint,
        device=infinicore.device("cpu"),
        dtype=override,
        cache_config=StaticKVCacheConfig(max_batch_size=1, max_cache_len=16),
    )
    assert engine.dtype == getattr(infinicore, expected)
    assert engine.hf_config["torch_dtype"] == expected
    # The public configuration alone could pass while native C++ still used
    # the checkpoint dtype. Inspect the actual native parameter metadata too.
    metadata = engine.state_dict()
    assert (
        metadata[0]["model.embed_tokens.weight"].dtype
        == getattr(infinicore, expected)._underlying
    )


@pytest.mark.parametrize("override", [None, "bfloat16"])
def test_model_runner_honors_dtype(checkpoint, override):
    config = EngineConfig(
        model_path=checkpoint,
        device="cpu",
        dtype=override,
        skip_load=True,
        cache_type="static",
        max_cache_len=16,
        enable_prefix_caching=False,
    )
    runner = ModelRunner(config, initialize_processor=False)
    try:
        expected = infinicore.float32 if override is None else infinicore.bfloat16
        assert runner.dtype == expected
        assert runner.model_engine.dtype == expected
    finally:
        runner.close()


def test_rejects_unsupported_dtype_before_native_init(checkpoint):
    with pytest.raises(ValueError, match="Unsupported model dtype"):
        InferEngine(checkpoint, device=infinicore.device("cpu"), dtype="int8")
