"""Qwen checkpoint loading and GPU model contracts used by MTP.

CPU loading tests run unconditionally. GPU checks require
INFINILM_QWEN_MTP_TEST_MODEL and optionally INFINILM_QWEN_MTP_TEST_TP=2.
"""

import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

import infinicore
import pytest
import torch
from infinilm.cache import PagedKVCacheConfig
from infinilm.distributed import DistConfig
from infinilm.infer_engine import InferEngine
from infinilm.llm.model_runner.mtp_runner import MTPRunner
from infinilm.modeling_utils import (
    _remap_qwen3_5,
    load_model_state_dict_by_file,
    load_state_dict,
)
from safetensors.torch import load_file, save_file


def test_mtp_is_opt_in_and_gemma_norms_are_converted():
    config = {"text_config": {"linear_key_head_dim": 2, "linear_num_key_heads": 1}}
    norms = [
        "model.language_model.norm.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
    ]
    gdn_norm = "model.language_model.layers.0.linear_attn.norm.weight"
    weights = {key: torch.tensor([0.25, -0.5]) for key in norms}
    weights[gdn_norm] = torch.tensor([1.0, 0.75])
    weights["mtp.fc.weight"] = torch.arange(8).reshape(2, 4).float()

    disabled = _remap_qwen3_5(dict(weights), config)
    assert not any(key.startswith("mtp.") for key in disabled)
    enabled = _remap_qwen3_5(dict(weights), {**config, "enable_mtp": True})
    for key in norms:
        torch.testing.assert_close(enabled[key], weights[key] + 1)
    for key in (gdn_norm, "mtp.fc.weight"):
        torch.testing.assert_close(enabled[key], weights[key])


@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.bfloat16])
def test_fp8_loading_preserves_scales_and_qkv_block_alignment(tmp_path, scale_dtype):
    prefix = "model.language_model.layers.0.linear_attn."
    weight = (
        torch.arange(1024 * 256)
        .reshape(1024, 256)
        .remainder(31)
        .float()
        .to(torch.float8_e4m3fn)
    )
    scales = (torch.arange(16).reshape(8, 2).float() / 1000 + 0.00012345).to(
        scale_dtype
    )
    path = tmp_path / "model.safetensors"
    save_file(
        {
            prefix + "in_proj_qkv.weight": weight,
            prefix + "in_proj_qkv.weight_scale_inv": scales,
        },
        path,
    )
    loaded = load_state_dict(str(path), preserve_fp8=True)
    config = {"text_config": {"linear_key_head_dim": 128, "linear_num_key_heads": 2}}
    mapped = _remap_qwen3_5(loaded, config)
    for name, start, end in (("q", 0, 256), ("k", 256, 512), ("v", 512, 1024)):
        key = prefix + "in_proj_" + name
        assert mapped[key + ".weight"].dtype == torch.float8_e4m3fn
        assert mapped[key + ".weight_scale_inv"].dtype == torch.float32
        torch.testing.assert_close(
            mapped[key + ".weight"].float(), weight[start:end].float(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            mapped[key + ".weight_scale_inv"],
            scales[start // 128 : end // 128].float(),
            rtol=0,
            atol=0,
        )


@pytest.fixture
def checkpoint():
    path = os.environ.get("INFINILM_QWEN_MTP_TEST_MODEL")
    if not path:
        pytest.skip("Set INFINILM_QWEN_MTP_TEST_MODEL to a tiny GPU checkpoint")
    return path


def create_model(path, enable_mtp=True):
    tp = int(os.environ.get("INFINILM_QWEN_MTP_TEST_TP", "1"))
    model = InferEngine(
        path,
        device=infinicore.device("cuda", 0),
        distributed_config=DistConfig(tp),
        cache_config=PagedKVCacheConfig(16, 64, 1),
        enable_mtp=enable_mtp,
        attention_backend="paged-attn",
    )
    load_model_state_dict_by_file(model, path, dtype=model.dtype)
    return model, tp


@pytest.fixture
def engine(checkpoint):
    # Malformed raw inputs can stop the workers: never share this engine across tests.
    return create_model(checkpoint)


def draft_inputs(model, hidden, ids, past=0):
    # from_torch currently assumes device 0 and contiguous storage.
    tensor = infinicore.strided_from_blob(
        hidden.data_ptr(),
        list(hidden.shape),
        list(hidden.stride()),
        dtype=infinicore.utils.to_infinicore_dtype(hidden.dtype),
        device=infinicore.device(hidden.device.type, hidden.device.index or 0),
    )
    runner = MTPRunner(
        SimpleNamespace(
            block_size=64,
            num_draft_tokens=1,
            tensor_parallel_size=len(model.distributed_config.tp_device_ids),
        ),
        model,
    )
    req = SimpleNamespace(block_table=[0], mamba_cache_index=1)
    return runner._inputs(req, ids, past, hidden=tensor)


def draft(model, hidden, ids, past=0):
    result = model.forward_raw(
        **draft_inputs(model, hidden, ids, past), sample_all_positions=True
    )
    logits = torch.empty(result["logits"].shape, dtype=hidden.dtype)
    infinicore.from_torch(logits).copy_(result["logits"])
    return logits


def test_tp_teardown_preserves_calling_device(checkpoint):
    if int(os.environ.get("INFINILM_QWEN_MTP_TEST_TP", "1")) < 2:
        pytest.skip("Communicator teardown requires TP2")
    torch.cuda.set_device(0)
    model, _ = create_model(checkpoint)
    previous = torch.cuda.current_device()
    del model
    gc.collect()
    assert torch.cuda.current_device() == previous


@pytest.mark.parametrize("strided", [False, True])
def test_cpu_and_each_tp_device_produce_identical_draft_logits(engine, strided):
    model, tp = engine
    width = model.hf_config["text_config"]["hidden_size"]
    hidden = torch.randn(1, 3, width, generator=torch.Generator().manual_seed(71))
    hidden = hidden.to(infinicore.utils.to_torch_dtype(model.dtype))
    expected = draft(model, hidden, [3, 8, 15])
    for rank in range(tp):
        device_hidden = hidden.to(f"cuda:{rank}")
        if strided:
            storage = torch.empty(
                1, 3, 2 * width, dtype=hidden.dtype, device=device_hidden.device
            )
            storage[..., ::2] = device_hidden
            device_hidden = storage[..., ::2]
        torch.cuda.synchronize(rank)
        actual = draft(model, device_hidden, [3, 8, 15])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # The next call must also reuse valid MTP history on every rank.
        actual_tail = draft(model, device_hidden[:, 2:], [3], past=3)
        draft(model, hidden, [3, 8, 15])
        expected_tail = draft(model, hidden[:, 2:].contiguous(), [3], past=3)
        torch.testing.assert_close(actual_tail, expected_tail, rtol=0, atol=0)


@pytest.mark.parametrize(
    "offsets,device_verify,error",
    [
        ([0, 3, 2], False, "checkpoint offsets must be increasing"),
        ([0, 1, 2], True, "one greedy"),
    ],
)
def test_invalid_checkpoint_boundaries_are_rejected(
    engine, offsets, device_verify, error, capfd
):
    model, _ = engine

    def i32(values):
        return infinicore.from_list(values, dtype=infinicore.int32)

    def i64(values):
        return infinicore.from_list(values, dtype=infinicore.int64)

    # Two Q=1 requests bypass prefill's offset validator. The checkpoint
    # validator must reject an out-of-range offset before indexing destinations.
    # Worker-side validation is logged before RankWorker.wait() reports shutdown;
    # acceptance-shape validation runs on the caller and raises its own message.
    raised_error = error if device_verify else "RankWorker stopped during run"
    with pytest.raises((ValueError, RuntimeError), match=raised_error):
        model.forward_raw(
            input_ids=i64([[3, 8]]),
            position_ids=i64([[0, 0]] * model.position_id_axes),
            past_kv_lengths=i32([0, 0]),
            total_kv_lengths=i32([1, 1]),
            input_offsets=i32(offsets),
            cu_seqlens=i32([0, 1, 2]),
            block_tables=i32([[0], [1]]),
            slot_mapping=i64([0, 64]),
            mamba_init_state_indices=i32([0, 0]),
            mamba_final_state_indices=i32([1, 2]),
            token_state_indices=i32([1, 2]),
            sample_all_positions=True,
            verify_draft=device_verify,
            top_k=1,
        )
    if not device_verify:
        captured = capfd.readouterr()
        assert error in captured.out + captured.err


@pytest.mark.parametrize("odd_vocab", [False, True])
def test_vocab_logits_and_local_argmax(checkpoint, tmp_path, odd_vocab):
    path = checkpoint
    if odd_vocab:
        original = Path(path)
        config = json.loads((original / "config.json").read_text())
        config["text_config"]["vocab_size"] += 1
        (tmp_path / "config.json").write_text(json.dumps(config))
        weights = load_file(original / "model.safetensors")
        for key in ("lm_head.weight", "model.language_model.embed_tokens.weight"):
            weights[key] = torch.cat([weights[key], weights[key][-1:]], dim=0)
        save_file(weights, tmp_path / "model.safetensors")
        path = str(tmp_path)
    model, tp = create_model(path)
    width = model.hf_config["text_config"]["hidden_size"]
    vocab = model.hf_config["text_config"]["vocab_size"]
    dtype = infinicore.utils.to_torch_dtype(model.dtype)
    hidden = torch.randn(1, 3, width, generator=torch.Generator().manual_seed(17)).to(
        dtype
    )
    inputs = draft_inputs(model, hidden, [3, 8, 15])
    full = model.forward_raw(**inputs, sample_all_positions=True)
    fast = model.forward_raw(**inputs, sample_all_positions=True, return_logits=False)
    assert (
        fast["output_ids"].to_numpy().tolist() == full["output_ids"].to_numpy().tolist()
    )
    last = model.forward_raw(**inputs, sample_all_positions=False, return_logits=False)
    assert (
        last["output_ids"].to_numpy().tolist()
        == full["output_ids"].to_numpy().tolist()[-1:]
    )

    # Reuse the same hidden vector to force maxima at shard boundaries, then
    # equal maxima on different ranks. The lowest global token ID must win.
    hidden_out = torch.empty(full["hidden_states"].shape, dtype=dtype)
    infinicore.from_torch(hidden_out).copy_(full["hidden_states"])
    weights = torch.zeros(vocab, width, dtype=dtype)
    for candidates in ((vocab - 1,), (vocab // tp - 1,), (1, vocab - 1)):
        weights.zero_()
        for token in candidates:
            weights[token] = hidden_out[0, -1]
        model.load_state_dict(
            {"lm_head.weight": infinicore.from_torch(weights)}, strict=False
        )
        full = model.forward_raw(**inputs, sample_all_positions=True)
        fast = model.forward_raw(
            **inputs, sample_all_positions=True, return_logits=False
        )
        assert (
            full["output_ids"].to_numpy().tolist()
            == fast["output_ids"].to_numpy().tolist()
        )
        assert int(fast["output_ids"].to_numpy()[-1]) == min(candidates)


def test_ordinary_qwen_head_matches_dense_projection_without_mtp(checkpoint):
    weights = load_file(Path(checkpoint) / "model.safetensors")["lm_head.weight"]
    reference = None
    for enabled in (True, False):
        model, _ = create_model(checkpoint, enable_mtp=enabled)
        runner = MTPRunner(
            SimpleNamespace(block_size=64, num_draft_tokens=1, tensor_parallel_size=1),
            model,
        )
        req = SimpleNamespace(block_table=[0], mamba_cache_index=1)
        inputs = runner._inputs(req, [3, 8, 15, 6], 0)
        full = model.forward_raw(**inputs, sample_all_positions=True)
        logits = torch.empty(full["logits"].shape, dtype=weights.dtype)
        infinicore.from_torch(logits).copy_(full["logits"])
        infinicore.sync_device()
        if enabled:
            hidden = torch.empty(full["hidden_states"].shape, dtype=weights.dtype)
            infinicore.from_torch(hidden).copy_(full["hidden_states"])
            infinicore.sync_device()
            # Independent, unsharded FP32 projection checks global vocabulary order.
            dense = torch.nn.functional.linear(hidden.float(), weights.float())
            torch.testing.assert_close(logits.float(), dense, atol=0.02, rtol=0.02)
            reference = logits
        else:
            torch.testing.assert_close(logits, reference, rtol=0, atol=0)
        fast = model.forward_raw(
            **inputs, return_logits=False, sample_all_positions=False
        )
        assert (
            fast["output_ids"].to_numpy().tolist() == logits[0, -1:].argmax(-1).tolist()
        )
        del model
        gc.collect()
