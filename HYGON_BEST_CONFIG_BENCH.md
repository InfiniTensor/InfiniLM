# Hygon DCU Best-Config Benchmark Guide (2026-04-28 update)

## 一、关键变化（必读）

DTK 2604 的 HSA loader 有一个 multi-rank JIT 死锁：当 ≥2 个 rank（同进程多线程）并发触发 `hsa_executable_freeze`（kernel 首次 JIT-load）时，`BlitKernel::SubmitLinearCopyCommand` 会卡在 DMA 完成信号上永不返回，所有后续 freeze 调用排队卡死。

**两个 workaround 各管一半场景，已经自动接好**：

1. **eager 模式（tp>1）**：`AMD_SERIALIZE_KERNEL=3` + `AMD_SERIALIZE_COPY=3`，强制 HIP runtime 序列化 kernel 启动和 DMA 拷贝，让 BLIT 引擎队列不再 race。`python/infinicore/__init__.py` 在没有 `--enable-graph` 的情况下自动 `setdefault`。代价：tp=1 eager 也变慢（55→35 tok/s decode）。

2. **graph 模式（tp>1）**：`Graph::instantiate` 的首轮 eager warmup 加 per-op cross-rank fence，让 4 个 rank 锁步 JIT-load 同一个 kernel（同 kernel 并发 freeze 没问题，只有发散 freeze 才会卡）。fence 通过 `PagedCompiler → context::stopGraphRecording → Runtime → GraphManager → Graph::instantiate` 一路传下去，由 `barrier_->wait()` 实现。**`AMD_SERIALIZE_KERNEL=3` 与 graph capture 冲突**（导致 `cudaError 900`），所以当 argv 里有 `--enable-graph` 时 auto-arm 主动跳过；fence 自己解决 JIT race，不需要 SERIALIZE。

argv 检测是启发式的；如果你是程序化启动 graph 模式（不走 CLI），需要显式 `INFINILM_HYGON_NO_SERIALIZE=1` 让 auto-arm 跳过。

## 二、跑通的配置矩阵（实测）

测试 prompt：`"How are you"`（默认）；`--temperature 1 --top-k 1`（贪心）；`--block-size 64`（gfx936 强制）；`--enable-paged-attn --attn=flash-attn`。

| 模型 | tp | graph | prefill tok/s | decode tok/s | status |
|---|---:|---|---:|---:|---|
| 9g_8b_thinking_llama | 1 | 否（eager） | 35.69 | 35.23 | OK |
| 9g_8b_thinking_llama | 1 | 是 | 139.24 | 56.44 | OK |
| 9g_8b_thinking_llama | 2 | 否 | 20.24 | 36.99 | OK（AMD_SERIALIZE） |
| 9g_8b_thinking_llama | 2 | 是 | **293.15** | **83.11** | OK（per-op fence） |
| 9g_8b_thinking_llama | 4 | 否 | 10.76 | 36.39 | OK（AMD_SERIALIZE） |
| 9g_8b_thinking_llama | 4 | 是 | **239.08** | **106.68** | OK（per-op fence） |
| FM9G_70B_SFT_MHA_qwen2 | 4 | 否 | 9.24 | 9.76 | OK（AMD_SERIALIZE） |
| FM9G_70B_SFT_MHA_qwen2 | 4 | 是 | **117.00** | **18.49** | OK（per-op fence） |
| 9g_8b_thinking_llama | 1 | 是 | server | server | OK（`/v1/chat/completions` 通） |
| 9g_8b_thinking_llama | 4 | 否 | server | server | **FAIL**（请求挂在 forward；offline tp=4 eager 同配置正常） |
| FM9G_70B_SFT_MHA_qwen2 | 4 | 是 | server | server | OK（需 `--num-blocks 64 --max-cache-len 1024`，否则 reset_cache OOM） |

**推荐配置**：
- 8B：`--enable-graph --tp 4`（106.7 tok/s decode，239 tok/s prefill，最快）
- 70B：`--enable-graph --tp 4`（18.5 tok/s decode，比 eager 1.9× 快；117 prefill 比 eager 12.7× 快）
- 单卡：8B tp=1 graph 仍是 56 decode。70B 单卡放不下 bf16 (151 GB)。

## 三、跑命令

### 0. 构建（已建好可跳过）

```bash
INFINILM_ENABLE_HYGON=1 INFINILM_BUILD_FLASH_ATTN=1 \
INFINILM_FLASH_ATTN_DIR=/abs/path/to/hygon/flash-attention \
pip install -e . --no-build-isolation

# pip 编辑安装走 pyproject 后端，不会触发 setup.py 的 cmake build hook，
# 改了 csrc 后要手动:
source /opt/dtk/env.sh && source /opt/dtk/cuda/cuda-12/env.sh
cmake --build build -j $(nproc)
```

### 1. 前置（每次跑前都做）

```bash
source /opt/dtk/env.sh
source /opt/dtk/cuda/cuda-12/env.sh
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH

# 卡都空闲？(tp>1 时尤其重要)
rocm-smi --showmeminfo vram --showuse | grep -E 'HCU\['
pgrep -af 'examples/.*\.py|inference_server' || echo 'no resident'
```

### 2. 8B tp=4 graph（推荐 — 最快 8B 路径）

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 python examples/bench.py \
  --device hygon --model /root/models/9g_8b_thinking_llama --tp 4 \
  --enable-paged-attn --enable-graph --attn flash-attn \
  --block-size 64 \
  --output-len 256 --max-new-tokens 256 \
  --temperature 1 --top-k 1
```

预期 decode ~107 tok/s。argv 里 `--enable-graph` 让 auto-arm 跳过 SERIALIZE。

### 3. 70B tp=4 graph（推荐 — 唯一 70B 配置）

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 python examples/bench.py \
  --device hygon --model /root/models/FM9G_70B_SFT_MHA_qwen2 --tp 4 \
  --enable-paged-attn --enable-graph --attn flash-attn \
  --block-size 64 \
  --output-len 256 --max-new-tokens 256 \
  --temperature 1 --top-k 1
```

预期：~46s 权重加载（盘 IO bound），生成 256 token ~14s（decode ~18.5 tok/s）。

### 4. 8B tp=1 graph（单卡推荐）

```bash
HIP_VISIBLE_DEVICES=0 python examples/bench.py \
  --device hygon --model /root/models/9g_8b_thinking_llama --tp 1 \
  --enable-paged-attn --enable-graph --attn flash-attn \
  --block-size 64 \
  --output-len 256 --max-new-tokens 256 \
  --temperature 1 --top-k 1
```

预期：~30s 总耗时，decode ≥ 50 tok/s。

### 5. server tp=1 graph (8B)

```bash
HIP_VISIBLE_DEVICES=0 python python/infinilm/server/inference_server.py \
  --device hygon --model /root/models/9g_8b_thinking_llama --tp 1 \
  --enable-paged-attn --enable-graph --attn flash-attn --block-size 64 \
  --max-batch-size 1 --host 127.0.0.1 --port 8001 &

# 等 "Application startup complete"，然后：
curl -sS -X POST http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"9g","messages":[{"role":"user","content":"How are you"}],"max_tokens":32}'
```

### 6. server tp=4 graph (70B) — 必须降低 cache pool 大小

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 python python/infinilm/server/inference_server.py \
  --device hygon --model /root/models/FM9G_70B_SFT_MHA_qwen2 --tp 4 \
  --enable-paged-attn --enable-graph --attn flash-attn --block-size 64 \
  --num-blocks 64 --max-cache-len 1024 \
  --max-batch-size 1 --host 127.0.0.1 --port 8002 &

curl -sS -X POST http://127.0.0.1:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"FM9G_70B_SFT_MHA_qwen2","messages":[{"role":"user","content":"How are you"}],"max_tokens":32}'
```

**关键**：`--num-blocks 64 --max-cache-len 1024`（默认 `--num-blocks 512 --max-cache-len 4096`）。70B 权重 sharded 后每 rank ~37 GB，64 GB DCU 剩 ~27 GB；默认 cache pool 在 `reset_cache` 阶段就 OOM（`pinnable_block_allocator.cc:86` `infinirtMalloc` failed）。bench.py 不踩这个坑是因为它按 `input_len + output_len` 动态算 `max_num_blocks`（"How are you" 只用 5 个 block），server 直接用 CLI 默认值。

**server tp=4 eager 仍不可用**（rank 0 单卡 16% util、其他 rank 闲置；offline tp=4 eager 通，定位中）。如果要 tp>1 server，强烈建议 `--enable-graph`。

## 四、铁律 / 易踩坑

1. **`--block-size 64` 必须显式传**。DTK fork 的 `paged_attention` kernel 在 gfx936 上对非 64 block_size 静默 half-write（每隔一个 bf16 槽位是 0），输出乱码不报错。`base_config.py` 在 `--device=hygon --attn=flash-attn` 时会强制校验，但 server 等其他入口不一定走那段。
2. **`--output-len 256 --max-new-tokens 256` 都要给**。前者是 bench 真正用来摆数据的 decode 长度，后者只是 per-request 上限；只给 max-new-tokens 会跑默认 20 token，数据没参考价值。
3. **改 csrc/ 后 `pip install -e .` 不会重编**。手跑 `cmake --build build -j$(nproc)` 才行。
4. **跑 70B tp=4 前确认 4 张卡都空闲**。70B 权重 ~140 GB，sharded 后每卡 ~37 GB；如果某卡有别的进程占着，会 OOM 或 hang。
5. **argv 里有 `--enable-graph` → auto-arm 跳过 SERIALIZE**。如果是程序化启动 graph 模式，显式 `INFINILM_HYGON_NO_SERIALIZE=1` 让 auto-arm 跳过；不然 graph capture 会 err 900 abort。

## 五、症状速查表

| 现象 | 原因 / 对策 |
|---|---|
| 输出每隔一汉字残缺 | `--block-size 64` 没传 |
| 只生成 20 token | `--output-len 256` 漏给 |
| tp>1 eager 卡在 "Processing cases: 0/1" | DTK loader race；确认 `AMD_SERIALIZE_*` 已自动 set（`infinicore/__init__.py`） |
| tp>1 + `--enable-graph` 抛 `cudaError 900` 或 `infiniopGemm Internal Error` | SERIALIZE_KERNEL 没跳过 → `--enable-graph` 必须在 argv 里，或显式 `INFINILM_HYGON_NO_SERIALIZE=1` |
| tp>1 + graph 在 `Graph::instantiate` 卡死 | per-op fence 没生效 → 检查 `csrc/engine/compiler/paged_compiler.cpp` 里 `stopGraphRecording([this](){barrier_->wait();})` 没退化成无参；`cmake --build build` 重编 |
| server tp=4 eager 请求 90s+ 不返回 | 已知问题（rank 0 单跑、其他 rank 闲置）；用 `--enable-graph` 替代 |
| 70B server tp=4 启动时 `cudaMalloc` Internal Error / `BlockAllocator::alloc failed` | cache pool 太大；加 `--num-blocks 64 --max-cache-len 1024`（每 rank ~37 GB 权重，默认 512 blocks 不够留 headroom） |
| 70B tp=4 卡 5+ 分钟没出 weights load | HCU 被别的进程占；`rocm-smi --showmeminfo vram` 排查 |

## 六、参考

- 调优历史 + diag dumps：`benchmark/opt_runs/`
- DTK loader bug + 修复：`benchmark/opt_runs/dtk_loader_race_findings.md`
- 实测 CSV：`benchmark/opt_runs/parity_matrix.csv`
