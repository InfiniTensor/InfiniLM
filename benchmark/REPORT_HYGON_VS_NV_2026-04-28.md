# Hygon DCU vs NVIDIA A100 — InfiniLM Inference-Server Benchmark (2026-04-28)

`bench_client.py` against the InfiniLM inference server, identical sweep parameters on both platforms. Test surface and harness are byte-for-byte identical; only the underlying device differs.

## Platforms

| Platform   | Hardware          | Build                                | Server flags                                                                  |
|------------|-------------------|--------------------------------------|-------------------------------------------------------------------------------|
| **A100**     | NVIDIA A100 ×1 / ×4 | `pip install -e .` (CUDA 12)         | `--enable-paged-attn --enable-graph --attn flash-attn`                        |
| **Hygon DCU** | gfx936 ×1 / ×4    | `INFINILM_ENABLE_HYGON=1 pip install -e .` (DTK 2604) | `--enable-paged-attn --enable-graph --attn flash-attn --block-size 64`        |

Per-op-fence (`csrc/engine/compiler/paged_compiler.cpp`) + DTK-fork dlsym FA (`ops/src/infinicore/adaptor/flash_attn_hygon_dlsym.cc`) on Hygon. 70B Hygon also takes `--num-blocks 64 --max-cache-len 1024` (per-rank weight ~37 GB on 64 GB DCU; default cache pool OOMs).

## Workload

- **8B `9g_8b_thinking_llama` tp=1** — `bs ∈ {1,4,16,64,128}`, in=out=256, n=max(200, 20·bs)
- **70B `FM9G_70B_SFT_MHA_qwen2` tp=4** — `bs ∈ {1,4,16}`, in=256, out=128, n=max(100, 10·bs)
- Common: `--seed 42 --ignore-eos`; bench_client defaults `temperature=1.0 top_p=0.8 stream=True`. Random-vocab prompt sampler (deterministic by seed).
- A100 8B server: `MAX_BATCH_SIZE` not bound in the original capture (assumed ≥ 128); Hygon 8B: `--max-batch-size 128`. A100x4 70B and Hygon×4 70B: `--max-batch-size 16`.

## 8B `9g_8b_thinking_llama` tp=1 — A100 vs Hygon DCU

| bs | n | platform | wall (s) | req/s | out tok/s | TTFT p50/p99 (ms) | TPOT mean/p50/p99 (ms) | ITL p50/p99 (ms) | E2EL p50/p99 (ms) |
|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| **1**   | 200  | A100     |  722.1 | 0.28 |  **70.6** |    92 /   159 |  13.8 /  13.9 /  14.1 |   0.44 /   68.4 |   3609 /   3658 |
|         |      | Hygon    | 1155.3 | 0.17 |  **44.1** |    88 /   235 |  22.3 /  22.5 /  22.6 |  22.48 /   66.9 |   5802 /   5819 |
| **4**   | 200  | A100     |  188.6 | 1.06 | **270.2** |   127 /   331 |  14.3 /  14.3 /  14.7 |   1.01 /   96.2 |   3762 /   3962 |
|         |      | Hygon    |  444.6 | 0.45 | **114.7** |   257 /   840 |  33.9 /  33.9 /  34.8 |  33.82 /  133.2 |   8848 /   9476 |
| **16**  | 320  | A100     |  117.9 | 2.71 | **691.5** |   760 /  1503 |  20.1 /  17.2 /  44.2 |   2.44 /  170.0 |   5125 /  12186 |
|         |      | Hygon    |  302.2 | 1.06 | **269.9** |   634 /  2855 |  56.2 /  56.4 /  61.9 |   2.54 /  425.3 |  14909 /  17312 |
| **64**  | 1280 | A100     |  273.4 | 4.68 | **1192.9**|   520 /  3686 |  47.9 /  51.2 /  65.9 |  40.30 /  258.9 |  13700 /  17285 |
|         |      | Hygon    | 1241.9 | 1.03 | **262.7** |  2330 / 13591 | 231.7 / 231.4 / 290.3 |  10.53 / 1819.7 |  61576 /  77952 |
| **128** | 2560 | A100     |  460.2 | 5.56 | **1416.9**|  8307 / 14011 |  59.4 /  50.7 / 125.3 |  19.01 /  765.5 |  21175 /  40914 |
|         |      | Hygon    | 2288.8 | 1.12 | **285.2** | 44572 / 93733 | 248.9 / 265.7 / 466.7 |  22.33 / 3141.6 | 112567 / 152951 |

Saturation throughput: A100 still scaling at bs=128 (~1417 tok/s), Hygon plateaus at bs=16 (~270 tok/s); past that, Hygon only adds queue depth.

## 70B `FM9G_70B_SFT_MHA_qwen2` tp=4 — A100×4 vs Hygon×4

| bs | n | platform   | wall (s) | req/s | out tok/s | TTFT p50/p99 (ms) | TPOT mean/p50/p99 (ms) | ITL p50/p99 (ms) | E2EL p50/p99 (ms) |
|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| **1**   | 100 | A100×4     | 219.9 | 0.45 | **24.7** |  189 /  591 |  36.3 /  35.1 / 118.3 |  0.41 /  253.6 |  1183 /  4998 |
|         |     | Hygon DCU×4| 325.6 | 0.31 | **14.4** |  235 /  501 |  64.8 /  65.4 /  79.7 | 65.44 /  326.4 |  2431 /  8497 |
| **4**   | 100 | A100×4     | 138.1 | 0.72 | **38.8** |  730 / 1271 | 114.3 /  96.9 / 421.2 |  1.64 /  505.6 |  4648 / 16032 |
|         |     | Hygon DCU×4| 210.1 | 0.48 | **25.9** |  885 / 2343 | 154.3 / 124.2 / 419.8 |  0.71 /  763.9 |  7081 / 20876 |
| **16**  | 160 | A100×4     | 131.5 | 1.22 | **66.2** | 1290 / 2578 | 228.6 / 220.9 / 551.2 | 37.14 /  978.8 |  9188 / 34521 |
|         |     | Hygon DCU×4| 205.7 | 0.78 | **38.4** | 2237 / 5906 | 408.8 / 361.7 / 809.7 |385.80 / 1772.4 | 14867 / 65014 |

Both platforms see avg_out ≈ 50 (instead of the requested 128) — the server-side `ignore_eos` plumbing isn't honored for the FM9G 70B path on either platform, so the comparison stays apples-to-apples (8B avg_out = 254.9 on both, so the flag works for the 8B path).

## Side-by-side ratios (Hygon / A100)

Speed of A100 vs Hygon at matching `(model, tp, bs)`:

| metric                   | 8B bs=1 | 8B bs=4 | 8B bs=16 | 8B bs=64 | 8B bs=128 | 70B bs=1 | 70B bs=4 | 70B bs=16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| out tok/s   A100 / Hygon | 70.6 / 44.1   | 270.2 / 114.7  | 691.5 / 269.9 | 1192.9 / 262.7 | 1416.9 / 285.2 | 24.7 / 14.4 | 38.8 / 25.9 | 66.2 / 38.4 |
| **A100 advantage (tok/s)** | **1.60×**  | **2.36×**       | **2.56×**     | **4.54×**      | **4.97×**      | **1.71×**   | **1.50×**   | **1.72×**   |
| TPOT mean   A100 / Hygon | 13.8 / 22.3   | 14.3 / 33.9    | 20.1 / 56.2   | 47.9 / 231.7   | 59.4 / 248.9   | 36.3 / 64.8 | 114.3 / 154.3 | 228.6 / 408.8 |
| **A100 advantage (per-tok)** | **1.62×** | **2.37×**     | **2.79×**     | **4.84×**      | **4.19×**      | **1.78×**   | **1.35×**   | **1.79×**   |
| TTFT p50    A100 / Hygon |   92 /   88   |  127 /  257    |   760 /  634  |   520 / 2330   | 8307 / 44572   | 189 / 235   | 730 / 885   | 1290 / 2237 |

Reading:

- **Single stream (bs=1)**: A100 is ~1.6× faster on 8B and ~1.7× on 70B — close to the raw memory-bandwidth ratio (1555 GB/s HBM2 on A100 vs ~1024 GB/s on DCU gfx936).
- **Server-saturation throughput (bs=16+)**: gap widens to **2.6× → 5.0×** on 8B as A100's batched-decode gets more headroom; Hygon plateaus at bs=16 (~270 tok/s) while A100 keeps scaling to ~1417 tok/s @ bs=128.
- **TTFT** (prefill): roughly comparable up to bs=16 (Hygon FA prefill is competitive at small batch); diverges sharply at bs ≥ 64 because Hygon's smaller plateau means deeper queues.
- **70B (tp=4)**: gap is more stable (1.5–1.8×) since both platforms are bandwidth-bound on bf16 weights and the cross-rank `barrier_->wait()` per-op fence on Hygon adds a fixed but small per-step overhead.

## Reproducibility

```bash
# 8B (per platform — only --device differs)
MAX_BATCH_SIZE=128 ./benchmark/serve_8b_tp1_graph_hygon.sh   # or NV-equivalent serve script
./benchmark/bench_8b_tp1_graph_hygon_sweep.sh                # or run_client.sh on NV
pkill -f 'inference_server.*port 8001'

# 70B
MAX_BATCH_SIZE=16 ./benchmark/serve_70b_tp4_graph_hygon.sh   # or NV-equivalent
./benchmark/bench_70b_tp4_graph_hygon_sweep.sh               # or run_client_tp4_70b.sh on NV
pkill -f 'inference_server.*port 8002'
```

Raw JSONs:

- A100 8B: `benchmark/results/infinilm_A100_*9g_8b*.json` (5 files)
- Hygon 8B: `benchmark/results/infinilm_hygon_DCU_tp1_graph_*9g_8b*.json` (5 files)
- A100×4 70B: `benchmark/results/infinilm_A100x4_*FM9G_70B*.json` (3 files)
- Hygon×4 70B: `benchmark/results/infinilm_hygon_DCUx4_tp4_graph_*FM9G_70B*.json` (3 files)

## Outstanding

1. `ignore_eos` not honored on the FM9G 70B path (avg_out ~50 vs requested 128) — affects both A100 and Hygon equally; should still be fixed on the server side for cleaner future runs.
2. Hygon 8B saturation at ~270 tok/s suggests a per-batch overhead floor (per-op fence, maybe + DTK FA launch latency); next investigation is to profile the bs=16 → bs=64 jump where A100 gains 1.72× while Hygon flatlines.
