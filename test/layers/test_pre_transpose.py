"""Check fused projections sharing a quantization object after weight packing."""

import json

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA device is required.")
def test_pre_transpose_and_reprocessing_preserve_logits(tmp_path):
    import infinicore
    from infinilm.cache import PagedKVCacheConfig
    from infinilm.infer_engine import InferEngine
    from infinilm.modeling_utils import load_model_state_dict_by_file
    from safetensors.torch import save_file

    config = {
        "model_type": "qwen2",
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "vocab_size": 128,
        "torch_dtype": "float16",
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 256,
        "rope_theta": 10000,
        "hidden_act": "silu",
        "eos_token_id": 0,
        "tie_word_embeddings": True,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    results, decode_results = [], []
    for packed in (False, True):
        engine = InferEngine(
            str(tmp_path),
            device=infinicore.device("cuda", 0),
            cache_config=PagedKVCacheConfig(8, 256, 1),
            attention_backend="paged-attn",
            pre_transpose=packed,
            enable_graph_compiling=True,
        )
        generator = torch.Generator().manual_seed(318)
        weights = {}
        for name, parameter in sorted(engine.state_dict()[0].items()):
            if name == "lm_head.weight":
                continue
            tensor = infinicore.Tensor(parameter)
            values = torch.randn(tensor.shape, generator=generator) * 0.03
            if "norm" in name and name.endswith("weight"):
                values.fill_(1)
            values = values.to(infinicore.utils.to_torch_dtype(tensor.dtype))
            weights[name] = values
        save_file(weights, tmp_path / "model.safetensors")
        load_model_state_dict_by_file(engine, str(tmp_path), dtype=engine.dtype)
        for recapture in (False, True):
            engine.process_weights_after_loading()
            output = engine.forward_raw(
                infinicore.from_list([[17, 83, 51]], dtype=infinicore.int64),
                position_ids=infinicore.from_list([0, 1, 2], dtype=infinicore.int64),
                past_kv_lengths=infinicore.from_list([0], dtype=infinicore.int32),
                total_kv_lengths=infinicore.from_list([3], dtype=infinicore.int32),
                input_offsets=infinicore.from_list([0, 3], dtype=infinicore.int32),
                cu_seqlens=infinicore.from_list([0, 3], dtype=infinicore.int32),
                block_tables=infinicore.from_list([[0]], dtype=infinicore.int32),
                slot_mapping=infinicore.from_list([0, 1, 2], dtype=infinicore.int64),
                sample_all_positions=True,
            )["logits"]
            copied = torch.empty(output.shape, dtype=torch.float16)
            infinicore.from_torch(copied).copy_(output)
            infinicore.sync_device()
            assert torch.isfinite(copied).all()
            results.append(copied)
            if recapture:
                # Recapture must preserve page zero of an ordinary model.
                engine.process_weights_after_loading()
            decoded = engine.forward_raw(
                infinicore.from_list([[23]], dtype=infinicore.int64),
                position_ids=infinicore.from_list([3], dtype=infinicore.int64),
                past_kv_lengths=infinicore.from_list([3], dtype=infinicore.int32),
                total_kv_lengths=infinicore.from_list([4], dtype=infinicore.int32),
                input_offsets=infinicore.from_list([0, 1], dtype=infinicore.int32),
                cu_seqlens=infinicore.from_list([0, 4], dtype=infinicore.int32),
                block_tables=infinicore.from_list([[0]], dtype=infinicore.int32),
                slot_mapping=infinicore.from_list([3], dtype=infinicore.int64),
            )["logits"]
            copied_decode = torch.empty(decoded.shape, dtype=torch.float16)
            infinicore.from_torch(copied_decode).copy_(decoded)
            infinicore.sync_device()
            decode_results.append(copied_decode)
        del engine
    for actual in results[1:]:
        torch.testing.assert_close(actual, results[0], atol=1e-3, rtol=1e-3)
    for actual in decode_results[1:]:
        torch.testing.assert_close(actual, decode_results[0], atol=1e-3, rtol=1e-3)
