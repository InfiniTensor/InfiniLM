# InfiniLM 重构后性能 Benchmark 报告

## 1. 测试目标

验证 `refactor/inline-infinicore` 分支重构后的 InfiniLM 在生产推理场景下的功能正确性与吞吐性能，作为重构验收的最终评估。

## 2. 测试环境

| 项 | 值 |
|---|---|
| 硬件 | 1× NVIDIA A100-SXM4-80GB (sm_80) |
| CUDA | 13.0 (Driver, nvcc) |
| cuDNN | 9.16.0-cuda-13 |
| NCCL | 2.28.9 |
| OS | Linux 5.15 (Ubuntu 22.04) |
| Python | 3.10.19 |
| PyTorch | 2.11.0+cu130 |
| InfiniLM | `refactor/inline-infinicore` @ `8a2e268` |
| flash-attention | vllm-project/flash-attention (FA2 路径，sm_80) |

## 3. 测试方法

### 3.1 服务端

`benchmark/server.sh` 启动 InfiniLM HTTP server（OpenAI-compatible），关键配置：

```
--device nvidia
--model /data-aisoft/mechdancer/models/9g_8b_thinking_llama
--tp 1
--cache-type paged --num-blocks 1024 --block-size 256
--max-new-tokens 4096 --max-batch-size 64
--enable-graph
--attn flash-attn
```

最大并发 64，KV cache 容量 = 1024 块 × 256 tok = 262 144 token。

### 3.2 客户端

`benchmark/bench_client.py`：基于 `httpx.AsyncClient` 的 SSE benchmark 客户端，与 `vllm bench serve` 输出指标等价：

- 协议：`POST /v1/chat/completions` `stream=true`，按 SSE chunk 时序计时
- 输入：固定 seed=42，按 random token id 解码生成 prompt（保证不同 batch 间 prompt 一致可比）
- 终止：`ignore_eos=true` 强制模型生成满 `max_tokens`，避免 EOS 提前结束影响吞吐
- 指标：TTFT、ITL、TPOT、E2EL（mean / p50 / p99 / min / max）+ request/output token throughput

### 3.3 扫描参数

| 维度 | 值 |
|---|---|
| 并发数 (max_concurrency) | 1, 4, 16, 64, 128 |
| input_len | 256 token |
| output_len | 256 token |
| 每并发 prompts | 20，下限 200 |

## 4. 结果

### 4.1 单卡 8B Llama 主表

| bs | 请求数 | 成功率 | TTFT mean (ms) | TTFT p99 (ms) | TPOT mean (ms) | E2EL mean (s) | req/s | **decode tok/s** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 200 | 200/200 | 94 | 159 | 13.85 | 3.6 | 0.28 | **70.6** |
| 4 | 200 | 200/200 | 137 | 331 | 14.31 | 3.8 | 1.06 | **270.2** |
| 16 | 320 | 320/320 | 798 | 1503 | 20.05 | 5.9 | 2.71 | **691.5** |
| 64 | 1280 | 1280/1280 | 1317 | 3687 | 47.90 | 13.5 | 4.68 | **1192.9** |
| 128 | 2560 | 2560/2560 | 7819 | 14011 | 59.44 | 22.9 | 5.56 | **1416.9** |

### 4.2 扩展性分析

**Decode 吞吐 vs 并发**：

```
       1     4    16    64   128
tok/s 71   270   692  1193  1417
ratio  -  3.83x 2.56x 1.72x 1.19x
```

| 并发跃迁 | 理论加速 | 实测加速 | 效率 |
|---|---|---|---|
| 1 → 4 | 4× | 3.83× | 96% |
| 4 → 16 | 4× | 2.56× | 64% |
| 16 → 64 | 4× | 1.72× | 43% |
| 64 → 128 | 2× | 1.19× | 60%（受限流） |

### 4.3 关键观察

**(1) 功能正确性 ✓**
4560 次请求 0 失败，输出连贯（端到端无回归）。

**(2) decode bound 转折点 = bs ≈ 16**
- bs ≤ 4：TPOT ≈ 14 ms（与单卡完全相同）→ 此区间是 latency-bound，加 batch 几乎免费
- bs = 16：TPOT 升到 20 ms，加 batch 开始有 cost
- bs ≥ 64：TPOT 翻倍到 48–60 ms，但 throughput 仍持续增长 → 进入 throughput-bound

**(3) bs=128 受 server `max-batch-size=64` 限流**
- TTFT p99 = 14 s（vs bs=64 的 3.7 s），多出来的 64 个请求被排队等待
- decode 吞吐仅多 19%，验证了 `max-batch-size` 是真实瓶颈
- 推荐生产配置：`max-batch-size` 与 `--max-concurrency` 对齐

**(4) flash-attn + paged + graph 组合稳定**
全程无 graph capture 错误、KV cache OOM 或 NCCL 报错。

## 5. 与重构前的对比

重构前 InfiniLM 依赖外部 InfiniCore（`$INFINI_ROOT/lib/`），同硬件同模型同参数下不可直接比较（构建系统差异、SO 装载路径差异）。仅就同一码树下：

- 编译 1×：从 `xmake build && xmake install` 改为 `pip install -e .`，无需预装 InfiniCore
- 部署后 SO 体积：原 `libinfiniop.so + libinfinirt.so + libinfiniccl.so + libinfinicore_cpp_api.so + libinfinicore_infer.so + 2× python 模块` = 7 个 SO；现合并为 `libinfinirt + libinfiniccl + libinfinicore + libflash-attn-nvidia + 2× python 模块` = 6 个（其中 flash-attn 可选，180 MB sm_80 single arch）
- 验证用例（单卡 + tp=4）输出 token 与重构前完全一致（语义等价）

## 6. 结论

1. **重构后 InfiniLM 在 CMake + flash-attn + paged + graph 全开下，单卡 A100-80GB 推理 8B Llama 达到 1417 tok/s decode（bs=128，但实际饱和点在 bs=64 ~1193 tok/s）**
2. **稳定性合格**：4560 个请求零失败
3. **bs=64 是当前 server 配置下的最优 throughput-latency 平衡点**：1193 tok/s decode，TTFT p99 < 4 s
4. **建议生产配置**：`--max-batch-size` 与负载并发对齐；`--num-blocks 1024` 对 in=out=256 场景有充足余量

## 7. TP=4 70B 分布式

### 7.1 配置

| 项 | 值 |
|---|---|
| 模型 | FM9G_70B_SFT_MHA_qwen2 (Qwen2ForCausalLM, 80 层, hidden=8192, attention heads=64, kv heads=64) |
| 硬件 | 4× A100-SXM4-80GB |
| TP 度 | 4 |
| dtype | bf16 |
| `--max-batch-size` | 16 |
| `--num-blocks` | 128（block_size=256，KV 容量 32 768 token） |
| `--max-new-tokens` | 1024 |
| 其他 | 同上：`--enable-graph --cache-type paged --attn flash-attn` |
| Output len | 128 token（70B 约 8 倍 8B 解码耗时，缩短以控制总测试时间） |

每卡显存占用 ~70 GB / 80 GB（35 GB 模型权重 + KV cache + 激活 + flash-attn workspace）。

### 7.2 主表

| bs | 请求数 | 成功率 | TTFT mean (ms) | TTFT p99 (ms) | TPOT mean (ms) | E2EL mean (s) | req/s | **decode tok/s** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 100/100 | 195 | 591 | 36.28 | 2.2 | 0.45 | **24.7** |
| 4 | 100 | 100/100 | 621 | 1271 | 114.28 | 5.5 | 0.72 | **38.8** |
| 16 | 160 | 160/160 | 1250 | 2578 | 228.57 | 12.9 | 1.22 | **66.2** |

### 7.3 观察

- **bs=1 decode 25 tok/s**：与 8B 单卡 bs=1（70.6 tok/s）对比，70B 是 8.75× 大但只慢 2.86×，说明 TP=4 把单 token 计算量做了 ~3× 加速（理论上限是 4×）
- **TPOT 在 bs=16 升到 229 ms**：4 卡 + 70B 大模型，decode 严重 memory-bound，加 batch 摊薄 KV cache 读取的收益有限
- **TTFT 比 8B 低很多**（bs=1 195 ms vs 8B 单卡 94 ms）：tp=4 把 prefill 时的 GEMM 切了 4 份并行，attention/QKV 都受益
- **稳定性**：360 次请求 0 失败，AllReduce 通信走 NCCL 全程无错

### 7.4 8B 单卡 vs 70B tp=4 对比（bs=1，每 token 计算量参考）

| 模型 | 参数量 | 硬件 | TPOT | decode tok/s | 计算量比 | 实测耗时比 |
|---|---|---|---|---|---|---|
| 8B Llama | 8 B | 1× A100 | 13.85 ms | 70.6 | 1× | 1× |
| 70B Qwen2 | 70 B | 4× A100 | 36.28 ms | 24.7 | 8.75× | 2.62× |

70B 在 tp=4 下每 token 实际只比 8B 单卡慢 2.6 倍，TP 加速比约 **3.34×**（理论 4×，效率 84%）— allreduce 通信 + tp 并行调度开销吃掉了 16% 的潜力。

## 8. 总结

| 验证项 | 结论 |
|---|---|
| 单卡功能 (8B llama) | ✓ 0 失败/4560 请求 |
| 分布式功能 (tp=4 70B) | ✓ 0 失败/360 请求 |
| 单卡饱和 throughput | 1193 tok/s @ bs=64（接近 server `max-batch-size` 限） |
| tp=4 70B throughput | 66 tok/s @ bs=16（受 KV cache 大小限制无法继续加 batch） |
| TP=4 加速比 | 3.34× / 4× = **84% 效率** |
| 路径稳定性 | flash-attn + graph + paged 三栈全程稳 |

## 9. 后续可扩展测试

- **input/output 长度扫描**：覆盖 in ∈ {32, 256, 4096}，out ∈ {256, 1024, 2048, 4096}
- **server `max-batch-size=128` 重跑**：测真实 bs=128 饱和吞吐（去除限流）
- **graph capture 关闭对照**：定量 graph 在 decode hot path 的实际收益
- **70B tp=2 / tp=8**：探索 TP 度的吞吐-效率最优点

## 附录

完整 raw JSON 在 `benchmark/results/`：
- `infinilm_A100_model=9g_8b_thinking_llama_bs={1,4,16,64,128}_in=256_out=256_n={200,200,320,1280,2560}_seed=42.json`

每个文件包含 `config`、`successful_requests`、`failed_requests`、`wall_time_s`、`request_throughput_per_s`、`output_token_throughput_per_s`、`total_output_tokens`，以及 `ttft_ms / itl_ms / tpot_ms / e2el_ms` 五项的 `{mean, p50, p99, min, max}`。

---

## 附录 B：Hygon DCU 适配启动流程

`server_hygon.sh` / `run_client_hygon.sh` / `run_client_tp4_70b_hygon.sh` 是 Hygon DCU 的对应脚本。设备号通过 `GPU=` 控制；脚本已固定 `HIP_VISIBLE_DEVICES` 与 `CUDA_VISIBLE_DEVICES` 同步设置，所以 `GPU=0,1,2,3` 即选 0–3 号卡。

### B.1 最佳启动配置（与 NVIDIA 主表对齐）

```
--device hygon                  # 走我们新建的 Hygon dispatch
--cache-type paged
--num-blocks 1024 --block-size 64    # block_size 必须是 64 的倍数（DTK flash-attn 限制）
--max-new-tokens 4096 --max-batch-size 64
--enable-graph                  # HIP graph capture，约 5× prefill 加速
--attn flash-attn               # dlsym 路径，从系统 flash_attn_2_cuda*.so 解析
```

构建走 `pip install -e .` 由 `setup.py` 自动 cmake：脚本第一次运行会触发，之后的运行是增量的。无需 `INFINIOPS_FLASH_ATTN_DIR`（Hygon 走 dlsym，不 link 自己的 `.so`）。

### B.2 单卡 8B（卡 0）

终端 1：
```bash
GPU=0 ./benchmark/server_hygon.sh
```

终端 2：
```bash
./benchmark/run_client_hygon.sh
```

### B.3 4 卡 tp=4 70B（卡 0,1,2,3）

终端 1（注：当前 `/root/models/FM9G_70B_SFT_MHA_qwen2/` 文件不完整 —— `model.safetensors` 只有 58 GB（应 ~140 GB），`tokenizer_config.json` 0 字节；运行前需补齐）：
```bash
GPU=0,1,2,3 \
TP=4 \
PORT=8103 \
MAX_BATCH_SIZE=16 \
NUM_BLOCKS=128 \
MAX_NEW_TOKENS=1024 \
MODEL=/root/models/FM9G_70B_SFT_MHA_qwen2/ \
./benchmark/server_hygon.sh
```

终端 2：
```bash
./benchmark/run_client_tp4_70b_hygon.sh
```

### B.4 已验证短跑（不走 server，直接 `examples/jiuge.py --device hygon`）

| Configuration                     | Prefill   | Decode   |
|-----------------------------------|-----------|----------|
| Eager static-cache                | 23 tok/s  | 28 tok/s |
| Eager `--enable-paged-attn`       | 51 tok/s  | 38 tok/s |
| Eager `--attn=flash-attn`         | 49 tok/s  | 38 tok/s |
| Graph paged-attn                  | 250 tok/s | 40 tok/s |
| **Graph flash-attn + paged**      | **256 tok/s** | **40 tok/s** |

所有路径在 `9g_8b_thinking_llama` 上 token-for-token 一致输出 `<think>Okay, the user is`，验证语义正确。

### B.5 单卡 8B Hygon DCU 主表（server-client 全 sweep）

测试配置：DCU 卡 0，server `--enable-graph --cache-type paged --attn flash-attn --num-blocks 1024 --block-size 64 --max-batch-size 64 --max-new-tokens 4096`，client in=256/out=256/seed=42。

| bs | 请求数 | 成功率 | TTFT mean (ms) | TTFT p50 (ms) | TTFT p99 (ms) | TPOT mean (ms) | E2EL mean (s) | req/s | **decode tok/s** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 200 | 200/200 | 156 | 136 | 417 | 34.56 | 8.9 | 0.11 | **28.5** |
| 4 | 200 | 200/200 | 411 | 359 | 907 | 37.91 | (~13) | 0.40 | **101.6** |
| 16 | 320 | 320/320 | 1958 | 1993 | 3058 | 63.07 | 18.0 | 0.89 | **226.9** |
| 64 | 1280 | 1280/1280 | 16157 | 16385 | 19884 | 154.49 | 55.4 | 1.15 | **294.4** |
| 128 | 2560 | 2560/2560 | 43236 | 37523 | 97069 | 272.11 | (~120) | 1.14 | **290.3** |

**4560 个请求 0 失败** ✅

#### 与 NVIDIA A100 主表对照

| bs | A100 tok/s | DCU tok/s | DCU/A100 | A100 TPOT | DCU TPOT | TPOT 比 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 70.6 | 28.5 | 0.40× | 13.85 ms | 34.56 ms | 2.50× |
| 4 | 270.2 | 101.6 | 0.38× | 14.31 ms | 37.91 ms | 2.65× |
| 16 | 691.5 | 226.9 | 0.33× | 20.05 ms | 63.07 ms | 3.15× |
| 64 | 1192.9 | 294.4 | 0.25× | 47.90 ms | 154.49 ms | 3.23× |
| 128 | 1416.9 | 290.3 | 0.20× | 59.44 ms | 272.11 ms | 4.58× |

#### 观察

1. **饱和点 bs=64** （tok/s 294，TTFT p99 < 20s），bs=128 因 server `max-batch-size=64` 限流反而略降到 290。与 NVIDIA 主表的拐点位置一致。
2. **小 batch 开销大**：bs=1 TPOT 比 NVIDIA 慢 2.5×，远超 HBM 带宽差异（A100 ~2 TB/s vs Hygon ~1.5 TB/s 只应慢 ~30%）。
3. **大 batch 比例恶化**：bs=64 慢 3.23×，bs=128 慢 4.58×。除了带宽差异，paged_attention 的 warp-kernel（DCU 没 wmma 用 CUDA-core）和 flash-attn 的 vllm_mha_varlen_fwd 在大 batch 下吃亏。
4. **稳定性**：4560 个 SSE 长链接零错，flash-attn dlsym + graph capture + paged-attn + RCCL（虽然 tp=1 没启用）三栈整体稳。

#### bs=1 深度分析（profiling-driven，校正了第 2 条的初步判断）

**结论：bs=1 的 25 ms TPOT 不是 host 端开销，是 GPU 自身**。基于 chrono microsecond 桩 + hipprof 数据：

| bucket | 平均 us / step | 占 TPOT % |
|---|---|---|
| `to_input` (Python args → C++ struct) | 6 | 0.02% |
| `get_compiled` (graph 查表 + 9 次 H2D copy) | 130 | 0.5% |
| `graph_run` (HIP graph launch 返回) | 1200 | 4.8% |
| **`sync` (GPU 实际计算)** | **23,400** | **94.7%** |
| **总计** | **24,800** | 100% |

GPU 端 compute 占了 94.7%，host 端总共只 1.4 ms。所以"DCU 单步固定开销大"这个 hypothesis 是错的 —— 真正的瓶颈在 GPU 内核执行本身。

**理论极限对照**：
- DCU HBM ~1.5 TB/s；8B BF16 ≈ 16 GB → 单步理论下限 10.7 ms
- DCU 实测 23.4 ms = 2.2× 理论 = **46% 带宽效率**
- A100 实测 13.85 ms = 1.7× 理论 = **58% 带宽效率**
- 目标 17.3 ms（80% A100）= 1.62× DCU 理论 = 需要比 A100 还高的带宽效率，**当前 kernel 配置下不可达**

**graph capture 已在工作**：通过 diag instrumentation 验证 prefill 走 EAGER（每次 prompt 长度可变，没法捕图），decode 全部走 GRAPH（每个 batch_size 都预捕了图）。HIP graph launch 仅占 4.8%，对比 NVIDIA cuGraphLaunch 的 ~0.02%，DTK 在 graph 调度上确实有 ~1 ms 的额外开销，但相对总 25 ms 不占主导。

要把 DCU bs=1 拉到 80% A100 的真正杠杆是 **GPU kernel 端**：
1. **Fuse rmsnorm + linear 减少 HBM 读出**（当前 rmsnorm 是独立 kernel，每次写出又读入）
2. **改写 attention 用 DCU 原生 mfma**（替代 paged_attention 的 warp/CUDA-core 实现，A100 用 wmma 占了相当一部分优势）
3. **量化（INT8 weights）**（HBM 流量降一半）—— 当前 InfiniLM Hygon 路径没有量化算子

#### bs=1 kernel-level 优化结果（custom GEMV，已落地）

后续基于 profile 数据继续推。GEMM 占 88% GPU 时间，单一 kernel `MT16x16x32` 是元凶 —— 它是 DTK hipBLAS heuristic 给 M=1 形 GEMM 选的 tile，运行在 ~10 GFLOPS（约 0.02% of peak）。

cuBLAS-compat shim 不暴露其它 algo（DTK 的 `cublasGemmAlgoTohipblasGemmAlgo` 只接受 `DEFAULT` 和 `DEFAULT_TENSOR_OP`），所以走"换 algo"路径无效。改写为 **purpose-built GEMV 内核**（`ops/src/infiniop/ops/gemm/nvidia/gemv_hygon.cuh`，64-thread wave-wide reduction，BF16 IO + FP32 accumulator）：

| 路径 | TPOT | tok/s | 相对 |
|---|---|---|---|
| 原 cuBLAS GEMM (MT16x16x32) | 25.14 ms | 39.78 | 1.00× |
| **custom GEMV** | **17.02 ms** | **58.74** | **1.48×** |
| **目标 (80% A100)** | **17.3 ms** | **56.5** | — |

✅ **达成目标**：58.74 ≥ 56.5 tok/s（83% of A100 70.6）。

集成位置：`gemm_nvidia.cu::calculate()` 在 `ENABLE_HYGON_API` 下，命中 `n=batch=1 + op_a=T + 标准 PyTorch 权重 layout (M×K row-major, lda=K)` 时走 GEMV，否则 fallthrough 到 cuBLAS（prefill / 大 batch 不受影响，TTFT 同 51 ms 完全一致）。

热路径形状（profile 抓的）：
- qkv_proj (fused): m=4608 k=4096
- o_proj: m=4096 k=4096
- gate/up_proj: m=4096 k=16384 ×2  (实际看到 m=4096, k=16384 应该是 down_proj 的 K 维)
- down_proj: m=4096 k=16384
- lm_head: m=32768 k=4096

GEMV 每输出一个 block，64 线程一个 wavefront，4-bf16-vector loads 沿 K 维步进。HBM-bandwidth-limited（46% → 估计 ~63% bandwidth efficiency 后），距离 A100 58% 已经更接近。

后续若要再压（往 90% A100），杠杆按收益排序：
1. **down_proj 大 K=16384**：每线程 256 元素，可改成 multi-block split-K + atomic reduce，降低 per-block 工作量
2. **lm_head 大 M=32768**：当前 32K 个 block 调度，可以 block 内多输出（M_per_block=4）减少 launch overhead
3. 算子融合（rmsnorm+gemv）：bs=1 下 HBM 节省小（~1%），优先级低

`max-batch-size` 调优（待跑） —— 不重新跑全 sweep，只跑 bs=1/64 的 max-batch∈{32,64,96}。

完整 raw JSON 在 `benchmark/results/infinilm_HygonDCU_model=9g_8b_thinking_llama_bs={1,4,16,64,128}_in=256_out=256_n={200,200,320,1280,2560}_seed=42.json`。

### B.6 70B tp=4 [失败 — 需要 RCCL/分布式调试]

70B 模型 151 GB 完整下载到位后做了三次尝试，全部失败。每次都从 `examples/jiuge.py --device hygon --model /root/models/FM9G_70B_SFT_MHA_qwen2 --tp N --backend cpp --max-new-tokens 8 --enable-paged-attn --attn flash-attn --block-size 64` 起手（直跑，不走 server）。

**显存数学**（70B BF16 = 140 GB 权重总量）：

| TP | 每卡权重 | 是否能 fit DCU 64 GB | 实测 |
|---|---|---|---|
| 1 | 140 GB | ❌ | 没试，必然 OOM |
| 2 | 70 GB | ❌ | OOM @ `cudaMalloc` from `pinnable_block_allocator.cc:100`，rank 0 init 阶段就挂 |
| 4 | 35 GB | ✅（含 KV cache + 激活后还有 ~25 GB 余量） | 见下两行 |

**TP=4 + `--enable-graph`**：所有 4 个 RankWorker 卡死在 `Graph::instantiate()` 内部录制阶段，hsa_signal_wait 无限等。GDB 抓到的栈包含 mha_kvcache::run + paged_attention（dlsym）、add_rms_norm、gemm、rms_norm —— 4 个 rank 各停在不同算子。RCCL proxy 线程（`ncclProxyService` / `ncclProxyServiceUDS`）还活着但没活跃通信。**HIP graph capture 与 RCCL 在 DTK 上似乎不兼容**。15 分钟后强 kill。

**TP=4 eager（去掉 `--enable-graph`）**：weight load 43.6 s ✓ → chat 模板 + "start generate" ✓ → **prefill 永远不结束**。10 分钟内只有 DCU 0 和 1 在 32% 利用率（226W），DCU 2 和 3 完全 idle（89W = 待机功耗）。栈显示 4 个 rank 在不同算子执行（一个在 rocblas GEMM 等 hsa_signal、一个在 mha_varlen launch、一个在 add_rms_norm、一个 launch 中），**desync 严重**，疑似 AllReduce ring 在 4-rank 拓扑上有问题，或者 vllm_mha_varlen_fwd 在多 rank 下死锁。

**bisect 后续 8B tp=4 同样的 hang**（用 8B 排除 70B 显存压力 / 模型大小因素）：
- `--tp 4` + `9g_8b_thinking_llama` + eager + `NCCL_DEBUG=INFO`：weight load 4.6 s ✓，RCCL 4-rank ring 建立成功（32 channels × P2P/direct pointer 全连通），nranks=4 nNodes=1 localRanks=4 ✓ → **prefill 仍然 hang**，pattern 为 1 个 rank 工作 / 3 个 rank idle（哪个 rank 工作每次跑随机）。
- 加 `HSA_FORCE_FINE_GRAIN_PCIE=1`（RCCL warn 提示的）：仍然 hang，pattern 不变。
- 工作的那个 rank 栈：`infinicore::op::distributed::AllReduce::run()` → `ncclAllReduce_impl` → `ncclLaunchKernel` → `hsa_signal_wait`，**死等 NCCL kernel 完成**。其它 3 ranks 还没到这个 allreduce，被卡在更早的 barrier。

**结论**：Hygon DCU 上 InferEngine 的 TP 路径有真实 deadlock bug（不是显存、不是 graph、不是模型规模），症状是 RCCL 多 rank desync 死锁。

bug 不是 RCCL 自身（ring 通了，P2P 通了），而是 InfiniLM 的 RankWorker 线程间协同。怀疑：
1. 各 rank 的 forward 路径通过 AllReduce 同步，但前后某个 op（最可能是 flash-attn dlsym 路径中的 ATen 调用，session_export.md §3 Bug 2b 的 stream-binding 问题在 tp=1 没暴露但 tp>1 暴露）改变了 stream 状态导致 desync
2. RankBarrier 实现可能有 bug，让某些 rank 先于其他通过

需要进一步专项调试（典型 1-2 天工作量）：
- 在 `csrc/engine/distributed/communication_group.cpp` 的 `AllReduce::run()` 加 entry/exit 日志，看 4 个 rank 是否都到达
- 在 `RankBarrier::wait()` 加日志看哪个 rank 不到
- 检查 `csrc/engine/rank_worker.cpp` 中 RUN command 的处理是否对所有 rank 同步进入 forward
- 验证 TP linear（`csrc/layers/...`）的 `ColumnParallelLinear`/`RowParallelLinear` 是否有 rank-specific 的 op 顺序差异

**Workaround**：当前 tp>1 不可用。8B 单卡 tp=1 全功能跑通（含 graph + flash-attn + custom GEMV，~58 tok/s），可作为生产最大配置。70B 必须等 tp 路径修复（或换更大显存的卡能 tp=1 跑下来 —— DCU 64 GB 不够）。

### B.7 TP>1 deadlock 解决 — 三个 DTK env var

之前所有 tp>1 hang 的根因都不在 InfiniLM 代码层，而在 **DTK runtime + RCCL 默认配置**。三个环境变量同时设置可解：

| Env | 作用 | 必要性 |
|---|---|---|
| `HSA_ENABLE_SDMA=0` | 禁用 SDMA，kernel 二进制上传走 shader-based copy | TP=2 即必须；不设则 `Segment::Freeze` 在 BlitKernel signal-wait 死循环 |
| `HSA_FORCE_FINE_GRAIN_PCIE=1` | RCCL 启动时显式 warn 提示的，强制 PCIe 细粒度内存协议 | TP>=2 推荐 |
| `NCCL_P2P_DISABLE=1` | 禁用 RCCL P2P，走共享内存 staging | TP>=4 必须；TP=2 不设也能跑 |

**实测 8B + flash-attn + paged-attn**：

| 配置 | 卡 | TTFT (ms) | Decode tok/s | 输出 |
|---|---|---|---|---|
| TP=2 (`HSA_ENABLE_SDMA=0` 即可) | 2,3 | 460 | **55.19** | `<think>Okay, the` ✓ |
| TP=4 (三个 env 全开) | 4,5,6,7 | 881 | **54.38** | `<think>Okay, the` ✓ |
| TP=4 70B (三个 env 全开) | 4,5,6,7 | 1006 | **14.59** | `您好，我是九格通用` ✓ |

70B 单卡 weight load 37 s（151 GB → 4-way 分片，每卡 ~38 GB）；前向走 RCCL ring，AllReduce 不再死锁。

**Workaround 已落到 `benchmark/server_hygon.sh`**：脚本里默认 `HSA_ENABLE_SDMA=0` + `HSA_FORCE_FINE_GRAIN_PCIE=1`，TP>1 时再追加 `NCCL_P2P_DISABLE=1`。用户也可以手动覆盖。

**残留问题**：
- 14.59 tok/s 在 70B 单 batch decode 看起来 communication-bound（PCIe ring 而非高速互联）；进一步压榨需要切到 NCCL P2P 或 NVLink 等价物，本机硬件不具备。
- 之前 GDB 抓到的"4 rank desync 在不同算子"那张栈，并不代表 InfiniLM 代码有 bug —— 实际上每个 rank 的 first-iter kernel 加载有不同的 lazy JIT 时长，加上没有 `HSA_FORCE_FINE_GRAIN_PCIE=1` 时 RCCL 的 sender/receiver 因 PCIe 细粒度 race 而 desync。三个 env 一开就消失。
- 旧版 §B.6 留作 archeology — 那时的诊断（"InferEngine TP 有 deadlock bug"）是错的，根因不在 InfiniLM 代码。
