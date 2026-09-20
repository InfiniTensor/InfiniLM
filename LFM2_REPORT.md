# InfiniLM LFM2-1.2B 适配报告

## 1. 项目内容

本项目在 InfiniLM 中新增 `LiquidAI/LFM2-1.2B` 支持。LFM2 不是普通的全 Attention 模型，其 16 个 Decoder 层由 10 个 ShortConv 层和 6 个全 Attention 层交错组成，因此除了 Attention KV Cache，还需要维护 ShortConv 的卷积状态。

本次实现包括：

- 注册 `model_type=lfm2`，将官方配置转换为 InfiniLM 使用的模型配置。
- 实现 LFM2 Decoder、RMSNorm、SwiGLU MLP 和长度为 3 的 gated depthwise ShortConv。
- 复用 Qwen3 GQA Attention，并按照官方 `layer_types` 组装混合 Decoder。
- 为 Static/Paged 两种缓存模式分配并路由 Attention KV Cache 和 ShortConv State Cache。
- 增加官方 Safetensors 权重名称到 InfiniLM 参数树的映射。
- 增加真实模型、同权重 tiny 模型、缓存/reset、低精度计算和性能记录脚本。

## 2. 实现思路

### 2.1 混合 Decoder

配置加载阶段根据 `full_attn_idxs` 生成每层的 `layer_types`。`full_attention` 层复用现有 Qwen3 Attention，`short_conv` 层使用新增的 `Lfm2ShortConv`。两类层共用 LFM2 RMSNorm 和 SwiGLU MLP，从而只新增 LFM2 特有结构，尽量复用现有基础设施。

### 2.2 ShortConv

ShortConv 首先通过 `in_proj` 生成三个分支 `B`、`C` 和 `x`，计算：

```text
y = out_proj(C * depthwise_causal_conv1d(B * x))
```

Prefill 阶段使用滑动窗口和 batched Matmul 实现长度为 3 的深度卷积；单 Token Decode 阶段读取每个请求对应的历史状态，只计算当前 Token，并把最后两个时间步写回 Conv State Cache。

低精度路径显式区分 Prefill 和 Decode 的舍入边界，以复现 Transformers 参考实现的 BF16 计算顺序。

### 2.3 双缓存与请求隔离

Attention 层继续使用现有 KV Cache。ShortConv 层单独分配 `[state_pool, hidden_size, kernel_size - 1]` 状态张量，通过请求的初始/最终状态索引读取和写回。Static Cache 预留零历史行，确保新请求不会读取上一个请求的卷积状态；Paged Cache 则按请求索引保存状态。

### 2.4 权重映射

官方 LFM2 的 Attention Norm、输出投影、FFN 和最终 Norm 名称与 InfiniLM 参数树不完全相同。`_remap_lfm2` 在加载时完成名称转换，同时保留 ShortConv 原有权重名称。真实模型权重映射已经完成闭环验证。

## 3. 复现流程

### 3.1 环境

主要 NVIDIA 验证环境：

```text
GPU: NVIDIA RTX 4090 24 GB
CUDA Toolkit: 12.8
Model: LiquidAI/LFM2-1.2B
Dtype: BF16
Decoding: greedy argmax
```

先按 InfiniCore README 编译并安装 NVIDIA 后端，并设置：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export INFINI_ROOT=/data/InfiniTensor/install/nvidia
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

然后构建并安装 InfiniLM：

```bash
xmake f -y -c -m release
xmake build -j4 _infinilm
xmake install _infinilm
python -m pip install --no-build-isolation --no-deps -e .
```

### 3.2 纯 Python 合同测试

```bash
PYTHONPATH=test/models/lfm2:python python -m unittest \
  test.models.lfm2.test_weight_remap \
  test.models.lfm2.test_state_routing \
  test.models.lfm2.test_short_conv_contract \
  test.models.lfm2.test_low_precision_contract \
  test.models.lfm2.test_run_config
```

当前结果：`16/16` 通过。

### 3.3 生成 Transformers 参考

分别对三个 Prompt 执行：

```bash
python test/models/lfm2/reference_lfm2_real.py \
  --model /path/to/LFM2-1.2B \
  --prompt "Who are you?" \
  --max-new-tokens 16 \
  --device cuda \
  --output artifacts/lfm2_transformers_cuda.json
```

另外两组 Prompt 为：

```text
请用一句中文介绍你自己。
Explain in three short points why recurrent state can reduce decoding work.
```

### 3.4 InfiniLM Static/Paged 验证

Static Cache：

```bash
python test/models/lfm2/run_infinilm_lfm2_real.py \
  --model /path/to/LFM2-1.2B \
  --device cuda \
  --cache-type static \
  --max-cache-len 256 \
  --max-new-tokens 16 \
  --repeat 2 \
  --reference artifacts/lfm2_transformers_cuda.json \
  --extra-reference artifacts/lfm2_transformers_cuda_zh.json \
  --extra-reference artifacts/lfm2_transformers_cuda_long.json \
  --output artifacts/lfm2_static_gate.json
```

Paged Cache 使用相同命令，将 `--cache-type` 改为 `paged`，并将 `--max-cache-len` 改为 `1024`。

## 4. 复现结果

### 4.1 NVIDIA RTX 4090

#### 2026-09-19 clean-build regression

为排除旧构建缓存或预置扩展对结果的影响，在一台新创建的 RTX 4090 24 GB 实例上进行了从源码开始的复验。该实例使用 CUDA 12.8、Python 3.12.3、PyTorch `2.6.0a0+ecf3bae40a.nv25.01`。构建前仅将随源码传输带入的旧 `.xmake` 缓存和旧 `_infinilm` 扩展改名保留；随后完整重编译了 `_infinilm` 的全部 C++ 单元并安装。编译耗时 195.193 秒，新扩展 SHA-256 为 `9c7a361464aaaa82826a46cbf12cb89d5f57779317a4301068429c0ea3202945`。

- tiny F32 Full/Prefill+Decode：设置 `NVIDIA_TF32_OVERRIDE=0` 后，最大 logits 绝对误差为 `5.960464477539063e-08`，小于 `1e-5`；最终 argmax 一致。
- Static KV Cache：`max_cache_len=256`，三种 Prompt 均与 Transformers 的 16 个 greedy token 精确一致；A/B/C/A/B/C 两轮结果一致。
- Paged KV Cache：`max_cache_len=1024`，三种 Prompt 均与 Transformers 的 16 个 greedy token 精确一致；A/B/C/A/B/C 两轮结果一致。

本次结果 JSON 已归档到 `work/artifacts/final_4090_20260919/`：`lfm2_native_tiny_cuda_strict_new4090.json`、`lfm2_final_static_256_new4090.json` 和 `lfm2_final_paged_1024_new4090.json`。

| 验收项 | Static | Paged |
|---|---:|---:|
| 三 Prompt Transformers 16-token 精确对齐 | 3/3 通过 | 3/3 通过 |
| A/B/C/A/B/C 重复请求 | 通过 | 通过 |
| 重复结果一致 | 通过 | 通过 |
| 208-token 输入、64 步固定长度压力测试 | 通过 | 通过 |
| Paged BF16 cache/full argmax 对照 | 不适用 | 56/56 |
| Static BF16 cache/full argmax 对照 | 54/56，未完全通过 | 不适用 |

Prompt `Who are you?` 的 16 个生成 Token 与 Transformers 一致：

```text
[550, 1283, 902, 14009, 6544, 16701, 5237, 811,
 1801, 7039, 916, 768, 6266, 3795, 803, 10003]
```

一次 RTX 4090 BF16 基线记录如下。计时范围只包括 engine forward 和设备同步，不包含模型加载、Tokenizer、元数据构造和输出回传，因此不是服务端端到端 TTFT：

| 缓存模式 | 208-token Prefill | 单 Token Decode 均值 | Decode tokens/s | 设备显存采样 |
|---|---:|---:|---:|---:|
| Static | 7.43 ms | 6.04 ms | 165.49 | 3096 MiB |
| Paged | 6.63 ms | 4.50 ms | 222.25 | 3096 MiB |

这些数据是单机基线，不是优化前后对比。

### 4.2 CPU

CPU 原生构建、LFM2 模型工厂、参数树和 tiny F32 Full/Prefill/Decode 已通过。当前最终源码尚未重新执行真实 1.2B CPU 全量回归，因此 CPU 不标记为真实模型完整支持。

### 4.3 昇腾 910B1

当前为部分支持：

- Ascend Runtime、设备复制和 LFM2 所需基础算子已完成验证。
- tiny LFM2 F32 Full 与 Prefill+Decode 一致。
- 真实 LFM2-1.2B 可以创建缓存、加载权重并进入第 0 层。
- 真实 BF16 在 ShortConv 的 Linear/GEMM 阶段仍会触发 CANN 同步错误，尚未获得可信 logits 和 Token 对齐结果。

因此本报告不把昇腾标记为端到端支持。相关 InfiniCore Ascend 实验代码也尚未达到可合并状态。

## 5. 已知限制

- 当前正确性主线限定 `batch_size=1`；多个不同长度请求的 packed Prefill 尚未实现 ShortConv 分段状态更新。
- tiny BF16 严格 logits 阈值仍未完全通过，虽然真实模型三 Prompt 的 greedy Token 已对齐。
- Static BF16 的 cache/full 对照为 54/56 argmax 一致，仍有两个低决策间隔步骤发生分叉；Paged 路径为 56/56。
- 尚未执行服务端压力测试、MMLU/C-Eval 和完整 CI。
- 昇腾真实 BF16 端到端推理尚未完成。

## 6. 平台状态结论

| 平台 | 状态 |
|---|---|
| NVIDIA RTX 4090 / CUDA 12.8 | 主要功能通过；Static/Paged 三 Prompt 生成与 Transformers 精确对齐 |
| CPU | 构建、模型创建和 tiny F32 通过；真实 1.2B 最终回归未执行 |
| Ascend 910B1 / CANN 9.0 | Runtime、基础算子、tiny F32 和模型加载通过；真实 BF16 Linear/GEMM 未完成 |
