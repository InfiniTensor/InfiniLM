#!/usr/bin/env python3
"""InfiniLM + KT (KTransformers) CPU-GPU heterogeneous MoE offload.

Performance-critical callback design (verified on L20, INT4 Q4_K_M):
  - Pre-allocated persistent GPU staging buffers (hidden/ids/weights/output)
  - Raw cudaMemcpyAsync via ctypes on the default stream (no per-call alloc)
  - Double-buffered output: layer N and N+1 use different buffers so the
    async C++ consumer (residual add) can never race the next snapshot
  - Output snapshot: KT reuses its internal output GPU buffer across calls,
    so we must copy() into our own buffer before returning to InfiniLM

Usage:
    model = LLM(model_path=..., cache_type="paged", attn_backend="paged-attn",
                enable_prefix_caching=False, ...)   # config.json needs use_kt_moe=true
    from infinilm.kt_integration import setup_kt_moe
    setup_kt_moe(model, model_path=GGUF_DIR, method="LLAMAFILE",
                 num_experts=512, num_experts_per_tok=10,
                 hidden_size=2048, moe_intermediate_size=512,
                 num_hidden_layers=48, num_gpu_experts=0,
                 max_tokens=ENGINE_MAX_NUM_BATCHED_TOKENS)
    outs = model.generate([...], ...)
    from infinilm.lib import _infinilm; _infinilm.clear_kt_moe_callbacks()

Requirements / sharp edges:
  - max_tokens must be >= the largest forward batch (prefill!), i.e. the
    engine's max_num_batched_tokens, NOT the decode batch size.
  - enable_graph must be False (CUDA graph capture executes this Python
    callback; capture would fail or replay stale outputs).
  - Models with GDN/linear-attention (qwen3_next): ensure num_blocks//4
    (mamba cache pool) >= max_batch_size or concurrency gets serialized.
  - Single GPU (device 0), tensor_parallel_size=1 only.
"""

import ctypes
import logging

import torch

logger = logging.getLogger(__name__)


def _load_cudart():
    """Load the CUDA runtime library, preferring versioned fallbacks."""
    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise OSError("libcudart not found (looked for libcudart.so/.so.12/.so.11.0)")


def setup_kt_moe(
    model,
    model_path,
    num_experts,
    num_experts_per_tok,
    hidden_size,
    moe_intermediate_size,
    num_hidden_layers,
    num_gpu_experts=0,
    method="LLAMAFILE",
    cpuinfer_threads=8,
    threadpool_count=1,
    max_tokens=512,
    chunked_prefill_size=512,
    moe_layer_freq=1,
    moe_layers=None,
):
    """Create one KTMoEWrapper per MoE layer and register InfiniLM callbacks.

    Args:
        model: the infinilm LLM object (used for config sanity checks).
        model_path: GGUF directory for KT expert weights (INT4 Q4_K_M recommended).
        num_gpu_experts: experts kept on GPU inside KT (0 = all experts on CPU).
        method: KT backend. "LLAMAFILE" (GGUF) is the verified path.
        cpuinfer_threads: CPU threads for KT compute pool.
        max_tokens: staging buffer capacity; MUST be >= engine max_num_batched_tokens
            (a larger prefill raises RuntimeError instead of corrupting memory).
        chunked_prefill_size: KT wrapper's chunked prefill size (its own buffers).
        moe_layers: explicit iterable of MoE layer indices (e.g. DSV2's
            range(first_k_dense_replace, num_hidden_layers)). Defaults to
            range(0, num_hidden_layers, moe_layer_freq).

    Returns:
        Number of layers registered.
    """
    import infinicore as ic
    from kt_kernel import KTMoEWrapper

    from infinilm.lib import _infinilm

    # ---- sanity checks -------------------------------------------------- #
    engine_cfg = getattr(model, "config", None)
    if engine_cfg is not None and getattr(engine_cfg, "enable_graph", False):
        raise RuntimeError(
            "KT offload is incompatible with enable_graph=True "
            "(CUDA graph capture would execute the Python KT callback)"
        )

    ne, topk, hs, mis = (
        num_experts,
        num_experts_per_tok,
        hidden_size,
        moe_intermediate_size,
    )
    if moe_layers is None:
        moe_layers = list(range(0, num_hidden_layers, moe_layer_freq))
    else:
        moe_layers = list(moe_layers)

    # ---- ctypes cudaMemcpyAsync on default stream (fast, no torch overhead) ----
    ca = _load_cudart()
    ca.cudaMemcpyAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    ca.cudaMemcpyAsync.restype = ctypes.c_int
    D2D = 1  # cudaMemcpyDeviceToDevice

    # ---- Persistent staging buffers shared by all layers (sequential access) ----
    h_buf = torch.empty(max_tokens, hs, dtype=torch.bfloat16, device="cuda")
    i_buf = torch.empty(max_tokens, topk, dtype=torch.int32, device="cuda")
    w_buf = torch.empty(max_tokens, topk, dtype=torch.float32, device="cuda")
    # Double-buffered output: consecutive layers alternate, so layer N+1's
    # snapshot can never overwrite memory layer N's async consumer still reads.
    out_bufs = [
        torch.empty(max_tokens, hs, dtype=torch.bfloat16, device="cuda")
        for _ in range(2)
    ]
    stream_cache = [None]  # resolved on first callback (worker thread)

    def make_cb(wr):
        def cb(hidden, weights, ids, layer):
            n = hidden.size(0)
            if n > max_tokens:
                raise RuntimeError(
                    f"KT staging overflow: forward batch {n} > max_tokens {max_tokens}. "
                    "Increase setup_kt_moe(max_tokens=...) to cover the engine's "
                    "max_num_batched_tokens (prefill batches), or lower "
                    "INFINILM_MAX_NUM_BATCHED_TOKENS."
                )
            if hidden.size(1) != hs:
                raise RuntimeError(
                    f"KT hidden size mismatch: got {hidden.size(1)}, expected {hs}"
                )
            # infinicore -> torch staging (async, default stream)
            rc = ca.cudaMemcpyAsync(
                h_buf.data_ptr(), hidden.data_ptr(), n * hs * 2, D2D, 0
            )
            if rc != 0:
                raise RuntimeError(f"KT cudaMemcpyAsync(hidden) failed: cudaError {rc}")
            rc = ca.cudaMemcpyAsync(
                i_buf.data_ptr(), ids.data_ptr(), n * topk * 4, D2D, 0
            )
            if rc != 0:
                raise RuntimeError(f"KT cudaMemcpyAsync(ids) failed: cudaError {rc}")
            rc = ca.cudaMemcpyAsync(
                w_buf.data_ptr(), weights.data_ptr(), n * topk * 4, D2D, 0
            )
            if rc != 0:
                raise RuntimeError(
                    f"KT cudaMemcpyAsync(weights) failed: cudaError {rc}"
                )
            if stream_cache[0] is None:
                stream_cache[0] = torch.cuda.current_stream().cuda_stream
            out = wr.forward(h_buf[:n], i_buf[:n], w_buf[:n], stream_cache[0])
            # KT reuses its output buffer between calls: snapshot before returning.
            ob = out_bufs[layer & 1][:n]
            ob.copy_(out)
            return ic.from_torch(ob)._underlying

        return cb

    # ---- Register per-layer wrappers; roll back on partial failure -------- #
    loaded = 0
    try:
        for layer_idx in moe_layers:
            mask = torch.tensor(
                [True] * num_gpu_experts + [False] * (ne - num_gpu_experts),
                dtype=torch.bool,
            )
            wr = KTMoEWrapper(
                layer_idx,
                ne,
                topk,
                hs,
                mis,
                mask,
                cpuinfer_threads,
                threadpool_count,
                model_path,
                chunked_prefill_size,
                method=method,
            )
            wr.load_weights(torch.arange(ne, dtype=torch.int32))
            _infinilm.set_kt_moe_callback(layer_idx, make_cb(wr))
            loaded += 1
    except Exception:
        # Never leave a partially-registered model behind: the C++ side would
        # fall through to a native path whose expert modules were skipped.
        _infinilm.clear_kt_moe_callbacks()
        raise

    logger.info(
        "KT MoE ready: %d layers [%d..%d], %d GPU + %d CPU experts, method=%s",
        loaded,
        moe_layers[0],
        moe_layers[-1],
        num_gpu_experts,
        ne - num_gpu_experts,
        method,
    )
    return loaded
