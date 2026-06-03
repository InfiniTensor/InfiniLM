# Copyright (c) 2025, InfiniCore
"""Hybrid compiled-prefill glue for InferEngine (torch iter-0 + C++ decode)."""

from __future__ import annotations

import logging
from typing import List, Optional

import infinicore
import torch

logger = logging.getLogger(__name__)


def hybrid_prefill_ready(engine) -> bool:
    """True when hybrid may run (CUDAGraph capture finished if enabled)."""
    runner = engine._compiled_prefill_runner
    if runner is None:
        return False
    from infinilm.compile.env import prefill_cudagraph_enabled

    if prefill_cudagraph_enabled() and not runner._cudagraph_capture_done:
        return False
    return True


def compiled_prefill_supported(engine) -> bool:
    if not engine._prefill_compile_enabled:
        return False
    if not engine.enable_paged_attn:
        return False
    if engine.hf_config.get("model_type") not in ("fm9g", "fm9g7b", "llama", "minicpm"):
        return False
    return True


def get_paged_kv_layers(engine):
    """Cached paged KV layer handles for share-weights hybrid prefill."""
    if engine._paged_kv_layers_cache is not None:
        return engine._paged_kv_layers_cache
    engine._paged_kv_layers_cache = [
        engine._tensor_from_blob_meta(meta)
        for meta in engine.paged_kv_blob_layers()
    ]
    return engine._paged_kv_layers_cache


def ensure_compiled_prefill_runner(engine, *, block_size: int = 256) -> None:
    if not compiled_prefill_supported(engine):
        return
    if engine._compiled_prefill_runner is not None:
        return

    from infinilm.compile import CompiledPrefillConfig, CompiledPrefillRunner
    from infinilm.compile.env import (
        compile_bucket_mode,
        compile_bucket_step,
        compile_max_seq_len,
        compile_warmup_seq_lens,
        prefill_cudagraph_enabled,
        prefill_share_weights_enabled,
    )
    from infinicore.utils import to_torch_dtype

    max_seq = compile_max_seq_len()
    cfg = CompiledPrefillConfig(
        model_path=engine._model_path,
        max_seq_len=max_seq,
    )
    torch_device = torch.device("cuda", 0)
    cpp_state_dict = None
    kv_layers = None
    if prefill_share_weights_enabled():
        cpp_state_dict = engine._cpp_state_dict_for_compile()
        if engine._paged_kv_layers_cache is not None:
            kv_layers = engine._paged_kv_layers_cache
        if engine.enable_paged_attn:
            cache_cfg = engine.get_cache_config()
            if cache_cfg is not None:
                block_size = cache_cfg.block_size()
    warmup_seq_lens = compile_warmup_seq_lens(max_seq)
    logger.info(
        "compiled prefill warmup seq_lens=%s mode=%s step=%s",
        warmup_seq_lens,
        compile_bucket_mode(),
        compile_bucket_step(),
    )
    runner = CompiledPrefillRunner(
        cfg,
        device=torch_device,
        dtype=to_torch_dtype(engine.dtype),
        warmup_seq_lens=warmup_seq_lens,
        cpp_state_dict=cpp_state_dict,
        kv_layers=kv_layers,
        block_size=block_size,
    )
    engine._compiled_prefill_runner = runner
    engine._compiled_prefill_ready = True
    if (
        kv_layers is not None
        and prefill_share_weights_enabled()
        and prefill_cudagraph_enabled()
    ):
        runner.ensure_cudagraph_capture(kv_layers, block_size=block_size)
    from infinilm.compile.mem_profile import snapshot_gpu_mem

    snapshot_gpu_mem("T3_server_idle")


def sample_token_id_from_logits(
    logits_1d: torch.Tensor,
    generation_config,
) -> int:
    """Match C++ ``random_sample`` on 1-D last-token logits; return scalar token id."""
    import random

    from infinicore.nn.functional import random_sample
    from infinilm.generation.utils import infini_to_numpy

    logits_infini = infinicore.from_torch(logits_1d.reshape(-1).contiguous())
    out = random_sample(
        logits_infini,
        random.random(),
        generation_config.top_p,
        generation_config.top_k,
        generation_config.temperature,
    )
    return int(infini_to_numpy(out).reshape(-1)[0])


def infinicore_input_ids_to_torch(input_ids) -> torch.Tensor:
    if not isinstance(input_ids, infinicore.Tensor):
        input_ids = infinicore.Tensor(input_ids)
    if input_ids.device.type != "cuda":
        input_ids = input_ids.to(infinicore.device("cuda", 0))
    return infinicore.to_torch(input_ids.contiguous())


def hybrid_compiled_prefill_step(
    engine,
    input_ids,
    *,
    input_ids_torch: Optional[torch.Tensor] = None,
    temperature: float,
    top_k: int,
    top_p: float,
    block_tables,
    slot_mapping,
    paged_block_size: int,
    cpp_prefill_kwargs: dict,
):
    """Iter-0 hybrid prefill: compiled torch logits (+ optional torch KV write)."""
    from infinilm.compile.env import prefill_share_weights_enabled

    ensure_compiled_prefill_runner(engine)
    runner = engine._compiled_prefill_runner
    if runner is None or not hybrid_prefill_ready(engine):
        infinicore.sync_device()
        return engine(**cpp_prefill_kwargs)

    if input_ids_torch is None:
        input_ids_torch = infinicore_input_ids_to_torch(input_ids)
    if prefill_share_weights_enabled():
        kv_layers = get_paged_kv_layers(engine)
        last_logits = runner.run_prefill_paged(
            input_ids_torch,
            kv_layers=kv_layers,
            slot_mapping=slot_mapping,
            block_size=paged_block_size,
        )
    else:
        last_logits = runner.run_prefill_last_logits(input_ids_torch)
        infinicore.sync_device()
        engine(**cpp_prefill_kwargs)

    infinicore.sync_stream()
    from infinilm.infer_engine import GenerationConfig

    gen_cfg = GenerationConfig(
        max_new_tokens=1,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    return sample_token_id_from_logits(last_logits, gen_cfg)


def try_hybrid_prefill_forward(engine, **model_input) -> Optional[List[int]]:
    """Server/scheduler prefill: hybrid torch compile when eligible (batch size 1)."""
    if not compiled_prefill_supported(engine) or not engine.enable_paged_attn:
        return None
    past_kv_lengths = model_input.get("past_kv_lengths")
    if past_kv_lengths is None or past_kv_lengths.shape[0] != 1:
        return None
    block_tables = model_input.get("block_tables")
    if block_tables is None:
        return None

    input_ids = model_input["input_ids"]
    seq_len = int(input_ids.shape[-1])
    from infinilm.compile.env import (
        compile_buckets,
        compile_max_seq_len,
        prefill_cudagraph_enabled,
        prefill_share_weights_enabled,
    )
    from infinilm.compile.runner import min_compiled_prefill_seq_len

    if not hybrid_prefill_ready(engine):
        return None

    min_seq = min_compiled_prefill_seq_len()
    if prefill_cudagraph_enabled():
        # C-Eval shorts: stay on C++ until smallest compile bucket (512).
        min_seq = max(min_seq, min(compile_buckets(compile_max_seq_len())))
    if seq_len < min_seq:
        return None

    paged_block_size = engine.get_cache_config().block_size()
    cpp_prefill_kwargs = {
        k: model_input[k]
        for k in (
            "input_ids",
            "position_ids",
            "past_kv_lengths",
            "total_kv_lengths",
            "input_offsets",
            "cu_seqlens",
            "block_tables",
            "slot_mapping",
            "temperature",
            "top_k",
            "top_p",
        )
        if k in model_input
    }
    logger.debug(
        "compiled prefill (server path, share_weights=%s), seq_len=%s",
        prefill_share_weights_enabled(),
        model_input["input_ids"].shape[-1],
    )
    token_id = hybrid_compiled_prefill_step(
        engine,
        model_input["input_ids"],
        input_ids_torch=model_input.get("input_ids_torch"),
        temperature=float(model_input.get("temperature", 1.0)),
        top_k=int(model_input.get("top_k", 1)),
        top_p=float(model_input.get("top_p", 1.0)),
        block_tables=block_tables,
        slot_mapping=model_input.get("slot_mapping"),
        paged_block_size=paged_block_size,
        cpp_prefill_kwargs=cpp_prefill_kwargs,
    )
    return [token_id]


def finish_cudagraph_capture_after_reset_cache(engine) -> None:
    """Run deferred CUDAGraph capture once paged KV tensors exist after reset_cache."""
    if engine._compiled_prefill_runner is None or not engine.enable_paged_attn:
        return
    from infinilm.compile.env import (
        prefill_cudagraph_enabled,
        prefill_share_weights_enabled,
    )

    if prefill_share_weights_enabled() and prefill_cudagraph_enabled():
        kv_layers = get_paged_kv_layers(engine)
        block_size = engine.get_cache_config().block_size()
        engine._compiled_prefill_runner.ensure_cudagraph_capture(
            kv_layers, block_size=block_size
        )
        logger.info(
            "compiled prefill: CUDAGraph capture complete (ready for hybrid prefill)"
        )
