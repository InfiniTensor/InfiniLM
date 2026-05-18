# Fused MoE microbench: Nsight Systems findings (vendor vs upstream)

This note summarizes **why** [`microbench_fused_moe_kernel.py`](microbench_fused_moe_kernel.py) reports a large **`cuda_ms_mean`** gap between **vendor** (`--impl infinilm`) and **upstream** (`--impl vllm`) on synthetic MiniCPM-like shapes (`E=32`, `N=384`, `H=768`, `top_k=4`, `num_tokens=128`, `bfloat16`, activation `silu`).

Profiling uses **Nsight Systems** with **`torch.cuda.nvtx`** ranges (`--nvtx` on the microbench) and optional wrapper [`run_microbench_fused_moe_nsys.sh`](run_microbench_fused_moe_nsys.sh).

## How to reproduce

Inside `minicpm5-moe` (pick a free GPU):

```bash
export CUDA_VISIBLE_DEVICES=1   # inside docker exec bash -lc '...'

IMPL=vendor  bash InfiniLM/examples/run_microbench_fused_moe_nsys.sh
IMPL=upstream bash InfiniLM/examples/run_microbench_fused_moe_nsys.sh
```

Artifacts (default):

- `InfiniLM/examples/bench_artifacts/nsys_microbench_moe_vendor.nsys-rep`
- `InfiniLM/examples/bench_artifacts/nsys_microbench_moe_upstream.nsys-rep`

CLI kernel summary:

```bash
nsys stats --report cuda_gpu_kern_sum InfiniLM/examples/bench_artifacts/nsys_microbench_moe_vendor.nsys-rep --timeunit ms | head -40
nsys stats --report cuda_gpu_kern_sum InfiniLM/examples/bench_artifacts/nsys_microbench_moe_upstream.nsys-rep --timeunit ms | head -40
```

In the GUI, filter the CUDA row by NVTX: **`microbench::infinilm::fused_experts`** vs **`microbench::vllm::fused_experts`**.

Numbers below are **illustrative** from one capture on **NVIDIA A100-SXM4-80GB**; re-run locally for authoritative percentages.

---

## Finding 1: Vendor GPU time spreads across generic PyTorch paths

Under NVTX `microbench::infinilm::fused_experts`, **most** accumulated GPU time in the summarized capture was **not** a single labeled Triton MoE GEMM kernel.

Dominant stacks included **generic ATen** / **libcudart CUB-style** kernels, for example (names abbreviated):

| Role (conceptual) | Examples seen in kernel summary |
|-------------------|--------------------------------|
| Indexing / gather | `index_elementwise_kernel`, `SelectSweepKernel`-style primitives |
| Fills / elementwise | `vectorized_elementwise_kernel` with `FillFunctor`, unary/binary elementwise |
| Concat / batched copies | `CatArrayBatchedCopy_*` |
| Reduction / compaction helpers | `DeviceReduceSingleTileKernel`, compaction init kernels |
| Copies | `unrolled_elementwise_kernel` / `direct_copy_kernel_cuda` |

The **named** **`fused_moe_kernel`** (Triton) appeared as a **minor** fraction of total GPU time in that capture (**on the order of a few percent** of the summed kernel table), implying that **routing of work, tensor layout transforms, indexing, and many small kernels** dominate the timeline relative to upstream for this harness.

Interpretation aligned with codebase direction: vendor path substitutes **pure PyTorch** for several vLLM CUDA pieces (`moe_align`, **`moe_sum` → `torch.sum`**, gated activation outside fused custom ops — see [`InfiniCore/python/infinicore/vendor/vllm_fused_moe/NOTICE`](../../InfiniCore/python/infinicore/vendor/vllm_fused_moe/NOTICE)).

---

## Finding 2: Upstream concentrates time in fused vLLM MoE + small dedicated kernels

Under `microbench::vllm::fused_experts`, **`fused_moe_kernel`** accounted for **the majority** of GPU time (order **50–60%** in the same style of capture).

Other significant kernels were **purpose-built MoE/aux** paths visible in naming:

| Kernel-style name (conceptual) | Role |
|--------------------------------|------|
| `fused_moe_kernel` | Main fused MoE matmul-heavy path |
| `moe_align_block_size_*` | Expert/token alignment aligned with upstream implementation |
| `act_and_mul_kernel` | Fused gated activation (e.g. SiLU × gate) vs separate Torch subgraph |
| `moe_sum_kernel` | Fuse top-k reductions vs `torch.sum` |

So upstream keeps **routing, activation, reduction, and matmul-heavy work** closer to **few specialized CUDA kernels**, which matches the measured **much lower `cuda_ms_mean`** on `.venv-vllm` vs system `python3` in [`microbench_fused_moe_kernel.py`](microbench_fused_moe_kernel.py).

---

## Finding 3: Cross-interpreter CUDA event deltas are directional; timeline is causal

[`run_moe_fused_stack_microbench_gap.sh`](run_moe_fused_stack_microbench_gap.sh) and the profiling memo already warn that **`cuda_ms_mean` across different PyTorch builds** is only loosely comparable.

Nsight nevertheless gives a **causal decomposition** inside each process:

- Vendor: correlate NVTX with **dense small-kernel bursts** vs sparse `fused_moe_kernel`.
- Upstream: correlate NVTX with **long `fused_moe_kernel` bars** plus short `moe_*` / `act_and_mul_*` tails.

---

## Practical follow-ups

1. **Reduce PyTorch-visible MoE scaffolding** toward upstream shapes: align/token packing parity, fused reduction and activation paths where acceptable for InfiniLM packaging.
2. **Tuned configs** for `(E, N, device)` if the goal is parity with tuned PyPI vLLM kernels (often missing exact JSON for `(32,384)` on A100 in bundled configs).
3. **Optional:** `nsys stats --report nvtx_gpu_proj_sum …` after `--trace=cuda,nvtx` to quantify time inside each `microbench::*` range.

---

## References

- Microbench driver: [`InfiniLM/examples/microbench_fused_moe_kernel.py`](microbench_fused_moe_kernel.py) (`--nvtx`)
- Nsys wrapper: [`InfiniLM/examples/run_microbench_fused_moe_nsys.sh`](InfiniLM/examples/run_microbench_fused_moe_nsys.sh)
- Vendor vs upstream hypotheses and backlog: [`InfiniLM/examples/MOE_VENDOR_UPSTREAM_CONCURRENCY_MEMO.md`](MOE_VENDOR_UPSTREAM_CONCURRENCY_MEMO.md)
- Broader profiling checklist: [`InfiniLM/examples/minicpm5_moe_inference_profiling.md`](minicpm5_moe_inference_profiling.md)
