"""Mamba-2 loading and request-state regression tests.

Set `INFINILM_MAMBA2_MODEL` to a prepared checkpoint for the device tests.
For strict FP32 comparisons, launch with `NVIDIA_TF32_OVERRIDE=0` on NVIDIA
or `INFINIOP_METAX_ALLOW_TF32=0` on MetaX.
"""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def preparation():
    path = Path(__file__).resolve().parents[3] / "scripts/prepare_mamba2_checkpoint.py"
    spec = importlib.util.spec_from_file_location("prepare_mamba2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TokenizerStub:
    eos_token_id = 0
    bos_token_id = 0
    pad_token_id = None

    def __len__(self):
        return 50277


def native_config():
    return {
        "d_model": 768,
        "n_layer": 24,
        "vocab_size": 50277,
        "ssm_cfg": {"layer": "Mamba2"},
        "pad_vocab_size_multiple": 16,
        "tie_embeddings": True,
    }


def test_native_checkpoint_contract(preparation):
    config = preparation.normalize_config(native_config(), TokenizerStub())
    assert config["vocab_size"] == 50288
    assert config["intermediate_size"] == 1536
    assert config["num_heads"] * config["head_dim"] == 1536
    assert config["n_groups"] == 1
    assert config["state_size"] == 128
    assert config["residual_in_fp32"]
    assert config["tie_word_embeddings"]
    assert config["eos_token_id"] == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"attn_layer_idx": [1]},
        {"d_intermediate": 128},
        {"ssm_cfg": {"layer": "Mamba1"}},
        {"ssm_cfg": {"layer": "Mamba2", "d_conv": 3}},
        {"ssm_cfg": {"layer": "Mamba2", "norm_before_gate": True}},
        {"ssm_cfg": {"layer": "Mamba2", "D_has_hdim": True}},
        {"ssm_cfg": {"layer": "Mamba2", "ngroups": 2}},
        {"ssm_cfg": {"layer": "Mamba2", "bias": True}},
        {"ssm_cfg": {"layer": "Mamba2", "conv_bias": False}},
        {"ssm_cfg": {"layer": "Mamba2", "d_state": 257}},
        {"d_model": 0},
        {"residual_in_fp32": False},
    ],
)
def test_unsupported_variants_are_rejected(preparation, changes):
    with pytest.raises(ValueError):
        preparation.normalize_config({**native_config(), **changes}, TokenizerStub())


def test_zero_padding_token_is_preserved(preparation):
    tokenizer = TokenizerStub()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 7
    assert preparation.normalize_config(native_config(), tokenizer)["pad_token_id"] == 0


def test_weight_mapping_preserves_state_parameters():
    from infinilm.modeling_utils import _remap_mamba2

    embedding = torch.randn(16, 8, dtype=torch.bfloat16)
    a_log = torch.tensor([-2.0, 1.5], dtype=torch.float16)
    dt_bias = torch.tensor([-3.1, -2.9], dtype=torch.float16)
    converted = _remap_mamba2(
        {
            "backbone.embeddings.weight": embedding,
            "backbone.layers.0.mixer.A_log": a_log,
            "backbone.layers.0.mixer.dt_bias": dt_bias,
        },
        {"tie_word_embeddings": True},
    )
    assert converted["lm_head.weight"] is converted["model.embeddings.weight"]
    assert converted["model.layers.0.mixer.A"].dtype == torch.float32
    torch.testing.assert_close(
        converted["model.layers.0.mixer.A"], -a_log.float().exp()
    )
    torch.testing.assert_close(
        converted["model.layers.0.mixer.dt_bias"], dt_bias.float()
    )


@pytest.fixture(scope="module")
def engine():
    model = os.environ.get("INFINILM_MAMBA2_MODEL")
    if not model:
        pytest.skip("Set `INFINILM_MAMBA2_MODEL` to run real-weight device tests.")
    import infinicore
    from infinilm.cache import PagedKVCacheConfig
    from infinilm.distributed import DistConfig
    from infinilm.infer_engine import InferEngine
    from infinilm.modeling_utils import load_model_state_dict_by_file

    instance = InferEngine(
        model,
        device=infinicore.device("cuda", 0),
        distributed_config=DistConfig(int(os.environ.get("INFINILM_MAMBA2_TP", "1"))),
        cache_config=PagedKVCacheConfig(64, 256, 4),
        enable_graph_compiling=os.environ.get("INFINILM_MAMBA2_GRAPH") == "1",
        attention_backend="paged-attn",
        pre_transpose=os.environ.get("INFINILM_MAMBA2_PRE_TRANSPOSE") == "1",
    )
    load_model_state_dict_by_file(instance, model, dtype=instance.dtype)
    return instance


def forward(engine, sequences, initial, final, past=None, *, sample_all_positions=True):
    import infinicore

    past = [0] * len(sequences) if past is None else past
    lengths = [len(sequence) for sequence in sequences]
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    ids = [token for sequence in sequences for token in sequence]
    positions = [
        position
        for start, length in zip(past, lengths)
        for position in range(start, start + length)
    ]

    def i32(values):
        return infinicore.from_list(values, dtype=infinicore.int32)

    result = engine.forward_raw(
        infinicore.from_list([ids], dtype=infinicore.int64),
        position_ids=infinicore.from_list(positions, dtype=infinicore.int64),
        past_kv_lengths=i32(past),
        total_kv_lengths=i32([p + n for p, n in zip(past, lengths)]),
        input_offsets=i32(offsets),
        cu_seqlens=i32(offsets),
        block_tables=i32([[0] for _ in sequences]),
        slot_mapping=infinicore.from_list([0] * len(ids), dtype=infinicore.int64),
        mamba_init_state_indices=i32(initial),
        mamba_final_state_indices=i32(final),
        sample_all_positions=sample_all_positions,
    )
    logits = result["logits"]
    copied = torch.empty(
        logits.shape, dtype=infinicore.utils.to_torch_dtype(logits.dtype)
    )
    infinicore.from_torch(copied).copy_(logits)
    infinicore.sync_device()
    return copied.float().reshape(-1, logits.shape[-1])


def test_prefill_continuation_and_decode(engine):
    import infinicore

    ids = torch.randint(
        1, 1000, (258,), generator=torch.Generator().manual_seed(19)
    ).tolist()
    full = forward(engine, [ids], [0], [1])
    first = forward(engine, [ids[:3]], [0], [2])
    middle = forward(engine, [ids[3:257]], [2], [2], [3])
    last = forward(engine, [ids[257:]], [2], [2], [257])
    continued = torch.cat([first, middle, last])
    assert torch.isfinite(continued).all()
    if engine.dtype == infinicore.float32:
        # The independent FP32 recurrence also changes slightly across GEMM shapes.
        torch.testing.assert_close(continued, full, atol=5e-4, rtol=3e-4)
    else:
        # Low-precision rounding changes raw logits, including a common offset.
        # Check the predictive distribution; the scan/state algebra is checked
        # separately in FP32 and against the independent operator reference.
        p, q = continued.log_softmax(-1), full.log_softmax(-1)
        mean_kl = (q.exp() * (q - p)).sum(-1).mean()
        is_bf16 = engine.dtype == infinicore.bfloat16
        assert mean_kl < (2e-2 if is_bf16 else 5e-4)
        agreement = (continued.argmax(-1) == full.argmax(-1)).float().mean()
        assert agreement >= (0.90 if is_bf16 else 0.98)


def test_tied_embeddings_share_device_storage(engine):
    for parameters in engine.state_dict():
        assert (
            parameters["lm_head.weight"].data_ptr()
            == parameters["model.embeddings.weight"].data_ptr()
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"rms_norm": False},
        {"rmsnorm": False},
        {"D_has_hdim": True},
        {"d_ssm": 128},
        {"d_intermediate": 128},
        {"attn_layer_idx": [1]},
        {"quantization_config": {"quant_method": "unsupported"}},
        {"dt_limit": [0, 1]},
        {"time_step_limit": [0, 1]},
        {"layer_norm_epsilon": 0},
        {"layer_norm_epsilon": -1e-5},
    ],
)
def test_cpp_rejects_unsupported_config_before_model_creation(
    preparation, tmp_path, changes
):
    import infinicore
    from infinilm.infer_engine import InferEngine

    config = {
        **preparation.normalize_config(native_config(), TokenizerStub()),
        **changes,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises((ValueError, RuntimeError)):
        InferEngine(str(tmp_path), device=infinicore.device("cpu", 0))


def test_cache_rebuild_preserves_weights_and_rebinds_state(engine):
    from infinilm.cache import PagedKVCacheConfig

    prompts = [[17, 83, 51], [142, 73, 6]]
    forward(engine, prompts, [0, 0], [1, 2])
    expected = forward(
        engine, [[91], [92]], [1, 2], [1, 2], [3, 3], sample_all_positions=False
    )
    for blocks in (48, 64):
        engine.reset_cache(PagedKVCacheConfig(blocks, 256, 4))
        test_tied_embeddings_share_device_storage(engine)
        forward(engine, prompts, [0, 0], [5, 3])
        actual = forward(
            engine, [[91], [92]], [5, 3], [5, 3], [3, 3], sample_all_positions=False
        )
        torch.testing.assert_close(actual, expected, atol=5e-4, rtol=3e-4)


def test_projection_remap_preserves_each_heads_inputs():
    from infinilm.modeling_utils import _remap_mamba2

    heads, dim, state, hidden = 4, 3, 2, 5
    weight = torch.randn(2 * heads * dim + 2 * state + heads, hidden)
    remapped = _remap_mamba2(
        {"backbone.layers.0.mixer.in_proj.weight": weight},
        {
            "num_heads": heads,
            "head_dim": dim,
            "state_size": state,
        },
    )
    x = torch.randn(7, hidden)
    original = torch.nn.functional.linear(x, weight)
    prefix = "model.layers.0.mixer."
    projected = torch.nn.functional.linear(
        x, remapped[prefix + "in_proj_zxd.weight"]
    ).reshape(7, heads, 2 * dim + 1)
    torch.testing.assert_close(
        projected[..., :dim].flatten(1), original[:, : heads * dim]
    )
    torch.testing.assert_close(
        projected[..., dim : 2 * dim].flatten(1),
        original[:, heads * dim : 2 * heads * dim],
    )
    torch.testing.assert_close(projected[..., -1], original[:, -heads:])
    for i, name in enumerate(("in_proj_b", "in_proj_c")):
        torch.testing.assert_close(
            torch.nn.functional.linear(x, remapped[prefix + name + ".weight"]),
            original[
                :, 2 * heads * dim + i * state : 2 * heads * dim + (i + 1) * state
            ],
        )


def test_decode_graph_reorders_requests_and_preserves_active_states(engine):
    if os.environ.get("INFINILM_MAMBA2_GRAPH") != "1":
        pytest.skip("Set `INFINILM_MAMBA2_GRAPH=1` for graph replay checks.")
    lengths = [3] * 5
    prompts = [[17 + i, 83, 51] for i in range(5)]
    forward(engine, prompts, [0] * 5, [1, 2, 3, 4, 5])
    forward(engine, prompts, [0] * 5, [6, 7, 8, 9, 10])
    for step in range(16):
        # Batch five exceeds the captured maximum and exercises device eager fallback.
        count = (1, 2, 4, 5)[step % 4]
        order = [(step + i) % 5 for i in range(count)]
        if step == 8:
            # Recompiling with live requests must preserve their recurrent states.
            engine.process_weights_after_loading()
        sequences = [[100 + step + i] for i in order]
        past = [lengths[i] for i in order]
        eager_rows, graph_rows = [i + 1 for i in order], [i + 6 for i in order]
        expected = forward(engine, sequences, eager_rows, eager_rows, past)
        actual = forward(
            engine, sequences, graph_rows, graph_rows, past, sample_all_positions=False
        )
        torch.testing.assert_close(
            actual, expected, atol=5e-4, rtol=3e-4, msg=f"Step {step}, batch {count}."
        )
        for i in order:
            lengths[i] += 1


def test_request_reordering_and_slot_reuse(engine):
    prompts = [[17, 83, 51], [142, 73, 6, 89, 13]]
    singles = [forward(engine, [prompt], [0], [1]) for prompt in prompts]
    packed = forward(engine, prompts[::-1], [0, 0], [5, 2])
    torch.testing.assert_close(packed[:5], singles[1], atol=8e-2, rtol=8e-2)
    torch.testing.assert_close(packed[5:], singles[0], atol=8e-2, rtol=8e-2)
    # A new request reads zero even when its destination contains another prompt.
    reused = forward(engine, [prompts[0]], [0], [5])
    torch.testing.assert_close(reused, singles[0], atol=8e-2, rtol=8e-2)


def test_processor_rejects_shared_writable_slots():
    from unittest.mock import patch

    from infinilm.processors.mamba2_processor import Mamba2Processor
    from infinilm.processors.mamba_processor import MambaProcessor

    processor = object.__new__(Mamba2Processor)
    requests = [SimpleNamespace(mamba_cache_index=1, num_local_cached_tokens=0)] * 2
    with patch.object(MambaProcessor, "build_model_inputs", return_value={}):
        with pytest.raises(RuntimeError, match="share"):
            processor.build_model_inputs(SimpleNamespace(scheduled_requests=requests))


@pytest.mark.parametrize(
    "options, message",
    [
        ({"cache_type": "static"}, "paged request-state"),
        ({"draft_model_path": "unused-draft"}, "rollback"),
        (
            {"kv_transfer_config": SimpleNamespace(kv_connector="unused-connector")},
            "state transfer",
        ),
        ({"enable_prefix_caching": True}, "Prefix caching"),
    ],
)
def test_unsupported_service_combinations_fail_before_loading(
    monkeypatch, options, message
):
    from infinilm.config.engine_config import EngineConfig
    from infinilm.llm import llm

    monkeypatch.setattr(llm, "read_hf_config", lambda _: {"model_type": "mamba2"})
    config = EngineConfig("unused-model", **{"enable_prefix_caching": False, **options})
    with pytest.raises((ValueError, RuntimeError), match=message):
        llm.LLMEngine(config)
