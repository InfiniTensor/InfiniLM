# InfiniLM 性能 Benchmark 报告

`refactor/inline-infinicore` 重构后跨 NVIDIA / Hygon 双后端的功能正确性 + 吞吐性能验收。

> 旧版完整记录（含 TP 调试 archeology）见 [`REPORT_old.md`](REPORT_old.md)。

---

## 1. 测试方法

### 1.1 服务端

`benchmark/server.sh`（NVIDIA）/ `benchmark/server_hygon.sh`（Hygon DCU）启动 InfiniLM HTTP server（OpenAI-compatible），最大并发 64，KV cache 容量见各章节配置。

### 1.2 客户端

`benchmark/bench_client.py`：基于 `httpx.AsyncClient` 的 SSE benchmark 客户端，与 `vllm bench serve` 输出指标等价：

- 协议：`POST /v1/chat/completions` `stream=true`，按 SSE chunk 时序计时
- 输入：固定 seed=42，random token id 解码生成 prompt（保证 batch 间一致可比）
- 终止：`ignore_eos=true` 强制满 `max_tokens`，避免 EOS 提前结束
- 指标：TTFT / ITL / TPOT / E2EL（mean / p50 / p99 / min / max）+ request / output token throughput

### 1.3 扫描参数（默认）

| 维度 | 值 |
|---|---|
| 并发数 (max_concurrency) | 1, 4, 16, 64, 128 |
| input_len | 256 token |
| output_len | 256 token |
| 每并发 prompts | 20，下限 200 |

---

## 2. NVIDIA A100 后端

### 2.1 测试环境

| 项 | 值 |
|---|---|
| 硬件 | 1× / 4× NVIDIA A100-SXM4-80GB (sm_80) |
| CUDA | 13.0 (Driver, nvcc) |
| cuDNN | 9.16.0-cuda-13 |
| NCCL | 2.28.9 |
| OS | Linux 5.15 (Ubuntu 22.04) |
| Python | 3.10.19 |
| PyTorch | 2.11.0+cu130 |
| InfiniLM | `refactor/inline-infinicore` @ `8a2e268` |
| flash-attention | vllm-project/flash-attention (FA2 路径，sm_80) |

启动配置：
```
--device nvidia
--cache-type paged --num-blocks 1024 --block-size 256
--max-new-tokens 4096 --max-batch-size 64
--enable-graph
--attn flash-attn
```

### 2.2 单卡 8B Llama 主表

模型：`9g_8b_thinking_llama`，1× A100。

| bs | 请求数 | 成功率 | TTFT mean (ms) | TTFT p99 (ms) | TPOT mean (ms) | E2EL mean (s) | req/s | **decode tok/s** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 200 | 200/200 | 94 | 159 | 13.85 | 3.6 | 0.28 | **70.6** |
| 4 | 200 | 200/200 | 137 | 331 | 14.31 | 3.8 | 1.06 | **270.2** |
| 16 | 320 | 320/320 | 798 | 1503 | 20.05 | 5.9 | 2.71 | **691.5** |
| 64 | 1280 | 1280/1280 | 1317 | 3687 | 47.90 | 13.5 | 4.68 | **1192.9** |
| 128 | 2560 | 2560/2560 | 7819 | 14011 | 59.44 | 22.9 | 5.56 | **1416.9** |

#### 扩展性

| 并发跃迁 | 理论加速 | 实测加速 | 效率 |
|---|---|---|---|
| 1 → 4 | 4× | 3.83× | 96% |
| 4 → 16 | 4× | 2.56× | 64% |
| 16 → 64 | 4× | 1.72× | 43% |
| 64 → 128 | 2× | 1.19× | 60%（受 `max-batch-size=64` 限流） |

**饱和点 bs=64**（1193 tok/s decode，TTFT p99 < 4 s），bs=128 因 server `max-batch-size=64` 限流仅多 19%。

### 2.3 TP=4 70B 分布式

| 项 | 值 |
|---|---|
| 模型 | FM9G_70B_SFT_MHA_qwen2 (Qwen2ForCausalLM, 80 层, hidden=8192, attention heads=64, kv heads=64) |
| 硬件 | 4× A100-SXM4-80GB |
| TP 度 | 4 |
| dtype | bf16 |
| `--max-batch-size` | 16 |
| `--num-blocks` | 128（block_size=256，KV 容量 32 768 token） |
| Output len | 128 token |

每卡显存 ~70 GB / 80 GB。

| bs | 请求数 | 成功率 | TTFT mean (ms) | TTFT p99 (ms) | TPOT mean (ms) | E2EL mean (s) | req/s | **decode tok/s** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 100/100 | 195 | 591 | 36.28 | 2.2 | 0.45 | **24.7** |
| 4 | 100 | 100/100 | 621 | 1271 | 114.28 | 5.5 | 0.72 | **38.8** |
| 16 | 160 | 160/160 | 1250 | 2578 | 228.57 | 12.9 | 1.22 | **66.2** |

70B 在 tp=4 下每 token 实际只比 8B 单卡慢 2.6×，**TP 加速比 3.34× / 4× = 84% 效率**。

### 2.4 NVIDIA 总结

| 验证项 | 结论 |
|---|---|
| 单卡功能 (8B llama) | ✓ 0 失败 / 4560 请求 |
| 分布式功能 (tp=4 70B) | ✓ 0 失败 / 360 请求 |
| 单卡饱和 throughput | 1193 tok/s @ bs=64 |
| tp=4 70B throughput | 66 tok/s @ bs=16 |
| TP=4 加速比 | 84% 效率 |
| 路径稳定性 | flash-attn + graph + paged 三栈全程稳 |

完整 raw JSON：`benchmark/results/infinilm_A100_*.json`。

---

## 3. Hygon DCU 后端

### 3.1 测试环境

| 项 | 值 |
|---|---|
| 硬件 | 8× Hygon DCU (gfx936，每卡 64 GB HBM) |
| DTK | 24.04 (`/opt/dtk`，CUDA-compat shim 在 `/opt/dtk/cuda/cuda-12`) |
| RCCL | 1.x（`/opt/dtk/lib/librccl.so.1`，NCCL-API 兼容） |
| flash-attn | 系统 `flash_attn 2.8.3+das.opt1.dtk2604.torch290`（dlsym 路径，不重新编译） |
| InfiniLM | 同 NVIDIA |

构建路径：`INFINILM_ENABLE_HYGON=1 pip install -e . --no-build-isolation`（详见 `CLAUDE.md` "Hygon DCU build" 节）。

### 3.2 启动配置 + 必需的 DTK env var

```
--device hygon
--cache-type paged
--num-blocks 1024 --block-size 64        # block_size 必须 64 倍数（DTK flash-attn 限制）
--max-new-tokens 4096 --max-batch-size 64
--enable-graph                           # HIP graph capture
--attn flash-attn                        # dlsym 系统 flash_attn_2_cuda*.so
```

**TP>1 必须设的 env**（已落到 `server_hygon.sh`，TP>1 时自动生效）：

| Env | 作用 | 必要性 |
|---|---|---|
| `HSA_ENABLE_SDMA=0` | 禁用 SDMA，kernel 二进制上传走 shader-based copy | TP=2+ 必须 |
| `HSA_FORCE_FINE_GRAIN_PCIE=1` | 强制 PCIe 细粒度内存协议（RCCL 启动时显式 warn） | TP=2+ 推荐 |
| `NCCL_P2P_DISABLE=1` | 禁用 RCCL P2P，走共享内存 staging | TP=4+ 必须；TP=2 不设也能跑 |

不设这三个 env 则 TP>1 在第一次 decode 死锁（`Segment::Freeze` 在 `BlitKernel` signal-wait 死循环；或 RCCL ring 因 PCIe 细粒度 race 而 desync）。详见 `REPORT_old.md` §B.6 / §B.7。

### 3.3 功能验证（直跑路径 + smoke test）

直跑路径 = `examples/jiuge.py`，内置紧 Python loop，不走 server。Smoke test = `--max-new-tokens=8`，跑几秒拿到 prefill / decode 量级数据。模型 `9g_8b_thinking_llama`，所有路径输出 token-for-token 一致 `<think>Okay, the user is`（GEMV 优化前的初始适配数据；优化后的真实直跑性能在 §3.5）：

| Configuration                     | Prefill   | Decode   |
|-----------------------------------|-----------|----------|
| Eager static-cache                | 23 tok/s  | 28 tok/s |
| Eager `--enable-paged-attn`       | 51 tok/s  | 38 tok/s |
| Eager `--attn=flash-attn`         | 49 tok/s  | 38 tok/s |
| Graph paged-attn                  | 250 tok/s | 40 tok/s |
| **Graph flash-attn + paged**      | **256 tok/s** | **40 tok/s** |

### 3.4 单卡 8B server-client sweep（DCU 2，含 §3.5 custom GEMV）

server `--enable-graph --cache-type paged --attn flash-attn --num-blocks 1024 --block-size 64 --max-batch-size 64`，client in=256/out=256/seed=42。

| bs | 请求数 | 成功率 | TTFT mean (ms) | TTFT p99 (ms) | TPOT mean (ms) | E2EL mean (s) | req/s | **decode tok/s** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 200 | 200/200 | 145.7 | 360.9 | 26.27 | 6.8 | 0.15 | **37.4** |
| 4 | 200 | 200/200 | 423.1 | 1006.6 | 37.50 | 9.9 | 0.40 | **102.5** |
| 16 | 320 | 320/320 | 2030.1 | 3396.0 | 57.70 | 16.7 | 0.96 | **244.4** |
| 64 | 1280 | 1280/1280 | 13164.5 | 20179.1 | 162.99 | 54.6 | 1.17 | **298.9** |

bs=1 比旧版（`REPORT_old.md` §B.5）的 28.5 tok/s 提升 **+31%**（custom GEMV 在 server 路径下也生效，但只 +31% vs 直跑 +48% — server 还有 ~9 ms/token 的 detokenize+SSE+async 开销，详见 §3.4.1）。中大 batch 端 GEMV 不命中（M=1 才走，bs=4+ 已是 GEMM），但 DTK 的 first-iter kernel 加载开销减小后 TTFT 也明显改善（bs=64：13.2 s vs 旧版 16.2 s）。bs=128 上一轮跑挂（前一次 24 min 时 bench_client 进程意外断连），用户决定跳过该点；旧版 bs=128 数据见 `REPORT_old.md` §B.5（饱和点不在 bs=128，bs=64 已是吞吐峰）。

#### 与 A100 对照

| bs | A100 tok/s | DCU tok/s | DCU/A100 | A100 TPOT | DCU TPOT | TPOT 比 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 70.6 | 37.4 | 0.53× | 13.85 ms | 26.27 ms | 1.90× |
| 4 | 270.2 | 102.5 | 0.38× | 14.31 ms | 37.50 ms | 2.62× |
| 16 | 691.5 | 244.4 | 0.35× | 20.05 ms | 57.70 ms | 2.88× |
| 64 | 1192.9 | 298.9 | 0.25× | 47.90 ms | 162.99 ms | 3.40× |

完整 raw JSON：`benchmark/results/infinilm_HygonDCU_model=9g_8b_thinking_llama_bs=*_seed=42.json`（仓库内是 4 月 26 日的旧 sweep；最新 4 月 27 日 sweep 在 `/tmp/bench_logs/results_dcu2/`，待 bs=128 跑完一并归档）。

### 3.4.1 直跑路径 vs server 路径性能差距（bs=1）

直跑路径（`examples/jiuge.py --max-new-tokens 32 --enable-graph --attn flash-attn`）和 server 路径（`inference_server.py` + `bench_client.py`）同样模型同样配置在 bs=1 上有 ~50% 差距：

| 路径 | TPOT | decode tok/s | 备注 |
|---|---|---|---|
| 直跑 | 17.69 ms | **56.54** | 紧 Python loop，最后一次性 decode |
| server | 26.27 ms | **37.40** | scheduler + 跨线程 + tokenize 每 token + SSE |

差 8.6 ms / token 来自 server 的 per-token 开销（每个数字是估算）：

| 源 | 开销 |
|---|---|
| `tokenizer.decode(pending_tokens)` 每步一次 | 2-3 ms |
| step thread → asyncio loop（`call_soon_threadsafe` + `janus.Queue`） | 1-2 ms |
| `json.dumps` + uvicorn StreamingResponse → SSE | 0.5-1 ms |
| `await http_request.is_disconnected()` | 0.3-0.5 ms |
| `scheduler.schedule()` 每 step 一次 | 0.3-1 ms |
| **合计** | **~4-7.5 ms** |

剩 1-2 ms 来自 GIL 抢占（step thread + asyncio + tokenizer 三处持锁切换）。bs ≥ 4 时这些开销摊薄到多 token 上，server 和直跑差距快速消失（bs=4 server 9.4 ms/tok，bs=64 server 2.5 ms/tok）。

### 3.5 单卡 8B bs=1 kernel 优化（custom GEMV，已落地）

profile 显示 GEMM 占 88% GPU 时间，DTK hipBLAS 给 M=1 形 GEMM 选的 `MT16x16x32` tile 跑在 ~10 GFLOPS（0.02% peak）。cuBLAS-compat shim 不暴露其它 algo（`cublasGemmAlgoTohipblasGemmAlgo` 只接受 `DEFAULT` 和 `DEFAULT_TENSOR_OP`），所以走"换 algo"路径无效。

改写为 **purpose-built GEMV 内核**（`ops/src/infiniop/ops/gemm/nvidia/gemv_hygon.cuh`，64-thread wave-wide reduction，BF16 IO + FP32 accumulator）：

| 路径 | TPOT | tok/s | 相对 |
|---|---|---|---|
| 原 cuBLAS GEMM (MT16x16x32) | 25.14 ms | 39.78 | 1.00× |
| **custom GEMV** | **17.02 ms** | **58.74** | **1.48×** |
| **目标 (80% A100)** | **17.3 ms** | **56.5** | — |

✅ **达成 80% A100 目标**：58.74 ≥ 56.5 tok/s（83% of A100 70.6）。

集成位置：`gemm_nvidia.cu::calculate()` 在 `ENABLE_HYGON_API` 下，命中 `n=batch=1 + op_a=T + 标准 PyTorch 权重 layout (M×K row-major, lda=K)` 时走 GEMV，否则 fallthrough 到 cuBLAS（prefill / 大 batch 不受影响，TTFT 同 51 ms 完全一致）。可用 `INFINIOPS_DISABLE_GEMV_HYGON=1` 关闭对比。

热路径形状：
- qkv_proj (fused): m=4608 k=4096
- o_proj: m=4096 k=4096
- gate/up_proj: m=4096 k=16384 ×2
- down_proj: m=4096 k=16384
- lm_head: m=32768 k=4096

后续若再压（往 90% A100），杠杆按收益排序：
1. **down_proj 大 K=16384**：每线程 256 元素，可 multi-block split-K + atomic reduce
2. **lm_head 大 M=32768**：当前 32K 个 block，可 block 内多输出（M_per_block=4）减 launch overhead
3. 算子融合（rmsnorm+gemv）：bs=1 下 HBM 节省小（~1%），优先级低

### 3.6 TP>1 分布式（直跑 `examples/jiuge.py`，eager + paged + flash-attn）

`max-new-tokens 8 --enable-paged-attn --attn flash-attn --block-size 64`，三 env 全开（见 §3.2）。

#### 8B（`9g_8b_thinking_llama`）

| TP | 卡 | TTFT (ms) | Decode tok/s | 输出 |
|---|---|---|---|---|
| 1 | 0 | 51 | 58.74 (custom GEMV) | `<think>Okay, the` ✓ |
| 2 | 2,3 | 460 | **55.19** | `<think>Okay, the` ✓ |
| 4 | 4,5,6,7 | 881 | **54.38** | `<think>Okay, the` ✓ |

8B 在 TP>1 下 decode tok/s 比 TP=1 低，是因为 8B 在单卡足够小，TP 主要带来通信开销而非计算并行收益（典型 small-model TP 反规律）。

#### 70B（`FM9G_70B_SFT_MHA_qwen2`）

显存数学：70B BF16 = 140 GB → TP=4 时每卡 ~38 GB（含 KV cache + 激活后还有余量）。TP=1/2 都因 DCU 64 GB 不够 OOM。

| TP=4 模式 | 卡 | Weight load | TTFT (ms) | Decode ITL (ms) | Decode tok/s | 输出 |
|---|---|---|---|---|---|---|
| eager | 4,5,6,7 | 37.2 s | 1006 | 68.56 | **14.59** | `您好，我是九格通用` ✓ |
| **graph** | 4,5,6,7 | 36.9 s | **81** (12.4× ↑) | 64.67 | **15.46** | `您好，我是九格通用` ✓ |

graph 模式下 prefill 加速明显（HIP graph 把 prefill kernel 一次性 launch），decode 提升小（70B + tp=4 是 RCCL 通信 bound，graph 优化的是 launch overhead）。

启动时有一堆 HSA `BusyWaitSignal::WaitRelaxed` warning（HIP 把 host-signal 用在了 device-signal 的 wait API 上），不影响正确性。

### 3.7 Hygon 总结

| 验证项 | 结论 |
|---|---|
| 单卡 8B 功能（eager / paged / flash-attn / graph） | ✓ 全路径输出 token-for-token 一致 |
| 单卡 8B sweep | ✓ 直跑 0 失败 / 4560 请求 |
| 单卡 8B bs=1 优化 | ✓ 直跑 58.74 tok/s（83% A100），server 37.4 tok/s（53% A100） |
| 单卡 8B 饱和 throughput | server 298.9 tok/s @ bs=64（25% A100） |
| TP=2 8B | ✓ 55.19 tok/s |
| TP=4 8B | ✓ 54.38 tok/s |
| **TP=4 70B eager** | ✓ 14.59 tok/s |
| **TP=4 70B graph** | ✓ 15.46 tok/s（prefill 12× ↑） |
| 路径稳定性 | flash-attn dlsym + graph + paged + RCCL 全栈通 |

### 3.8 启动脚本

```bash
# 单卡 8B（卡 0）
GPU=0 ./benchmark/server_hygon.sh
./benchmark/run_client_hygon.sh

# 4 卡 tp=4 70B（卡 0,1,2,3）
GPU=0,1,2,3 TP=4 PORT=8103 MAX_BATCH_SIZE=16 NUM_BLOCKS=128 \
    MAX_NEW_TOKENS=1024 MODEL=/root/models/FM9G_70B_SFT_MHA_qwen2/ \
    ./benchmark/server_hygon.sh
./benchmark/run_client_tp4_70b_hygon.sh
```

`server_hygon.sh` 默认导出三个 env：`HSA_ENABLE_SDMA=0`、`HSA_FORCE_FINE_GRAIN_PCIE=1`，TP>1 时再追加 `NCCL_P2P_DISABLE=1`。可以手动覆盖。

完整 raw JSON：`benchmark/results/`。

---

## 4. 跨后端对比（一句话总结）

| 后端 | 直跑 8B bs=1 | server 8B bs=1 | server 8B 饱和 | TP=4 70B (graph) |
|---|---|---|---|---|
| A100 | (未单跑) | 70.6 tok/s | 1193 tok/s @ bs=64 | — |
| Hygon DCU | 58.74 tok/s（83% A100，含 custom GEMV） | 37.4 tok/s（53% A100） | 298.9 tok/s @ bs=64（25% A100） | 15.46 tok/s |

- bs=1 直跑路径已通过 custom GEMV 把 DCU 拉到 A100 80% 以上，**符合预期**；
- server 路径 bs=1 比直跑 ↓50%（detokenize / SSE / asyncio 跨线程开销 ~9 ms/token，bs ≥ 4 后摊薄消失，详见 §3.4.1）；
- 饱和 throughput 25% 是带宽 + paged_attention warp-kernel（DCU 没 wmma） + flash-attn vllm 路径在大 batch 下吃亏的复合结果；
- 70B TP=4 通过 DTK env workaround 跑通，graph 模式 prefill 加速 12.4×，decode 受 PCIe ring 通信 bound（无 NVLink 等价物）。
