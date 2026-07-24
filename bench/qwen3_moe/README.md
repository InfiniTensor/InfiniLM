# Qwen3-MoE 并发吞吐基准（MetaX）

T2-1-3 的服务吞吐基准，对齐赛题「测试风格参考长文本优化赛题（T2-1-2）」的要求：
起 InfiniLM 服务 + `vllm bench serve` 压测 **concurrency × input-len × output-len** 矩阵，
产出 **base vs this** 的输出吞吐 / 总吞吐对比（赛题「性能」维度的打分依据）。

平台：沐曦 MetaX C500，TP=2。

## 文件

| 文件 | 作用 |
|---|---|
| `serve.sh` | 起 OpenAI 兼容服务（Qwen3-30B-A3B-Thinking，metax，TP=2，paged-attn + flash-attn + ignore-eos）|
| `run_bench.sh` | 客户端 `vllm bench serve` 遍历矩阵（**需装 vllm**）|
| `load_client.py` | **零依赖**并发压测客户端（纯标准库，无需 vllm/aiohttp）——vllm 装不上时用这个 |
| `summarize.py` | 解析结果文件成表格；给两个目录则输出 base-vs-this 的 Δ% |

> 两个客户端产出**同一格式**的结果文件（`bench_results/<TAG>/bsX_inY_outZ.txt`），
> summarize.py 都能解析。**MetaX 容器一般没有 vllm**（且与 maca torch 易冲突），推荐用 `load_client.py`。

## 服务依赖

服务端只需轻量依赖：

```bash
pip install fastapi uvicorn
```

## 用 load_client.py（推荐，无 vllm）

```bash
# 终端 A：起服务（见下）
# 终端 B：
python3 bench/qwen3_moe/load_client.py --tag this      # 或 --tag base
python3 bench/qwen3_moe/summarize.py bench_results/this
```

矩阵/并发/模型名均可用 `--batch-sizes/--input-lens/--output-lens/--model/--base-url` 覆盖。
`load_client.py` 用非流式请求测**输出吞吐 / 总吞吐**（赛题打分项）；不测 TTFT/TPOT（需流式，表里显示 `-`）。
输入长度用 CJK 填充字近似（~1 token/字）；要精确可加 `--tokenizer <模型路径>`（需 transformers）。

## 默认压测矩阵

- concurrency：`1 8 32`
- input-len：`32 256 2048`
- output-len：`256 1024`

（均可用环境变量覆盖，见各脚本头部注释。）

## base vs this 怎么跑

「this」= T2-1-3 分支（grouped_gemm 批化 MoE）。「base」= 改造前的**朴素逐 token×逐 expert** MoE
（`main` 分支 / grouped_gemm 之前的实现）。两者用**同一套客户端脚本**分别压测，只是换部署的 `.so`。

### 1) 跑 this（当前 T2-1-3）

```bash
# 终端 A：起服务（保持前台）
cd /data/InfiniLM
MODEL=/data/huggingface_home/Qwen3-30B-A3B-Thinking-2507 \
  bash bench/qwen3_moe/serve.sh
# 等打印 “load weights over” + uvicorn 起监听

# 终端 B：等服务 ready 后压测
cd /data/InfiniLM
curl -s http://127.0.0.1:8102/health && echo " <- server ready"
TAG=this bash bench/qwen3_moe/run_bench.sh
```

### 2) 跑 base（朴素 MoE 基线）

```bash
# 切到基线实现并重建 + 部署 .so（示例，按你的基线 commit 调整）
cd /data/InfiniLM && git stash && git checkout main
cd /data/InfiniCore && xmake install
cd /data/InfiniLM && xmake build _infinilm
cp build/linux/x86_64/release/_infinilm.cpython-310-x86_64-linux-gnu.so python/infinilm/lib/

# 起服务 + 压测（同上，改 TAG=base）
MODEL=/data/huggingface_home/Qwen3-30B-A3B-Thinking-2507 bash bench/qwen3_moe/serve.sh   # 终端 A
TAG=base bash bench/qwen3_moe/run_bench.sh                                                # 终端 B
```

> 注：base 与 this 必须用**完全相同的客户端参数**（矩阵、seed 策略、ignore-eos）才可比。
> 脚本已固定这些；两次只改 `TAG` 和部署的 `.so`。

### 3) 汇总对比

```bash
python3 bench/qwen3_moe/summarize.py bench_results/base bench_results/this
```

输出每个 (bs,in,out) 的 `out_tok/s` 与 `total_tok/s` 的 base→this 及 Δ%。
只看单侧：`python3 bench/qwen3_moe/summarize.py bench_results/this`。

## 显存 / 参数提示

- `NUM_BLOCKS` 默认 1024（≈ 26 万 token 的 KV，覆盖 concurrency 32 × (2048+1024) 有余）。
  **OOM 就调小**（`NUM_BLOCKS=512`），**报 KV 不足就调大**。
- `MAX_CACHE`（--max-cache-len）需 ≥ 最大 input+output（默认 4096 覆盖 2048+1024）。
- **注意力后端 `ATTN` 默认 `paged-attn`**：MetaX build **未编 flash-attn**（用 `--attn flash-attn`
  会报 "FlashAttention is not enabled in this build" 并崩 forward）。若 paged-attn 在你的 build 也不可用，
  退回 static cache：`PAGED=0 ATTN=default bash bench/qwen3_moe/serve.sh`（并发批处理能力会下降）。
- `--enable-graph` 默认**关**：MoE 前向含数据相关的 host dispatch + syncStream，graph capture 可能不兼容。
  想试开：`GRAPH=1 bash bench/qwen3_moe/serve.sh`，若起不来或结果异常就关掉。
- 采样固定 `top-k 1`（greedy）+ `ignore-eos`，保证每请求生成满 output-len、结果可比。

## 关注指标

赛题打分看 **Output token throughput (tok/s)** 与 **Total Token throughput (tok/s)**；
小规模（bs=1）作为「不回退」门槛。summarize.py 已抽取这两项 + req/s + TTFT + TPOT。
