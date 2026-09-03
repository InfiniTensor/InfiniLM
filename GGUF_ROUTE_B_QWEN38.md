# InfiniLM 适配 Qwen3.8-27B GGUF 技术报告

> 路线：GGUF 原生块量化（Route B）
> 目标模型：Qwen3.8-27B-UD-Q6_K
> 目标框架：InfiniLM + InfiniCore
> 文档日期：2026-09-03
> 状态：核心适配已完成并可运行；严格逐 token 一致性优化仍有可选提升空间

## 1. 摘要

本工作完成了 Qwen3.8-27B GGUF 模型到 InfiniLM 的原生块量化适配。这里的“原生”是指：

- GGUF 中的 Q8_0、Q4_K、Q5_K、Q6_K 权重块不先完整反量化为 BF16；
- 打包时直接保留 GGUF block bytes，并以 `torch.uint8` 张量写入 safetensors；
- 推理时由新增的 `linear_gguf` 算子在 GPU kernel 内按块解码并参与矩阵乘；
- 小 batch/decode 使用寄存器 GEMV，大 batch/prefill 使用分块解码加 GEMM；
- 暂不支持原生执行的少量权重在打包阶段显式转为 BF16，不允许静默回退。

最终产物能够完成全量 27B 模型加载、prefill、逐 token decode 和确定性生成。打包模型包含
947 个张量、6 个 safetensors 分片，总权重体积 23.264 GiB；其中 491 个张量保持 GGUF
block bytes，456 个张量为 BF16。

功能层面，GGUF 适配已经跑通。以 llama.cpp 为参照进行 32 个样例、每例 32 个 token 的严格
比较，当前接受的严格基线达到 **27/32 个样例完全一致、920/1024 个 token 一致**。这个指标
衡量的是两个不同推理后端的逐 token 数值复现程度，不等同于模型能否正确运行。原计划中的
`>=29/32` 属于额外的严格一致性优化目标，目前尚未达到，也不是 GGUF 适配可用性的必要条件。

## 2. 背景、目标与非目标

### 2.1 输入和输出

源模型：

```text
/home/liuxd/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K.gguf
```

当前正式打包产物：

```text
/home/liuxd/models/Qwen3.8-27B-GGUF-native-v2
```

源 GGUF 文件约 21.97 GB。打包后的 v1/native-v2 产物采用 InfiniLM 可装载的 safetensors
目录结构，同时在 `config.json` 中保存每个权重的 GGML 类型和 GGUF 量化配置。

### 2.2 主要目标

1. 在 InfiniLM 中加载并运行 Qwen3.8-27B GGUF 模型。
2. 尽可能保留 GGUF 原生量化块，避免将全部权重展开为 BF16。
3. 支持 Q8_0、Q4_K、Q5_K、Q6_K 四种主要 GGUF block 类型。
4. 同时覆盖 prompt prefill 和 autoregressive decode。
5. 建立可复用的 GGUF 打包、类型路由、算子和验证框架，以便后续适配其他模型。
6. 用独立门禁证明没有错映射、错 shape、错字节布局或静默稠密回退。

### 2.3 当前非目标

- 不要求 InfiniLM 与 llama.cpp 在所有输入上逐 bit 或逐 token 完全相同；
- 不在本阶段实现所有 IQ 系列 GGUF 量化格式；
- 不在本阶段实现 GGUF blob 的 tensor parallel 切分；
- 不把实验性 Q8 激活量化或局部 F32 路径默认启用；
- 不以 llama.cpp 的速度数据替代 InfiniLM 自身性能测试。

## 3. 模型特点与适配难点

Qwen3.8-27B 不是只包含标准全注意力层的简单 Transformer。模型共有 64 个 decoder layer，
其中包含全注意力层和 Gated DeltaNet/线性注意力层，并维护额外的 GDN/SSM state。适配难点主要
来自以下方面：

1. GGUF 张量命名与 InfiniLM 参数命名不一致；部分 GGUF 融合权重需要拆成多个运行时权重。
2. 不同 GGUF 类型具有不同 block size 和字节布局，U8 张量的第二维不是逻辑输入维度。
3. GGUF 某些二维权重的物理取向与运行时线性层的逻辑取向不同。
4. 线性注意力 `out_proj` 的 V head 布局需要额外的 grouped-to-tiled 置换。
5. GGUF 与原始模型在 RMSNorm gain 表达约定上存在差异，错误处理会造成系统性数值偏差。
6. prefill 的 M 较大，不能只实现单 token GEMV；同时又不能把完整权重永久展开为 BF16。
7. BF16 舍入、归约顺序、采样语义会使接近的 logits 在两个后端产生不同 top-1 token。

## 4. 总体技术路线

完整数据流如下：

```text
Qwen3.8-27B GGUF
        |
        v
gguf_mapping.py 生成唯一映射计划
        |
        v
gguf_to_infinilm.py
  |-- 支持类型：原始 block bytes -> U8 weight_bytes
  |-- 例外类型：显式解码 -> BF16 weight
  |-- 写入 ggml_types / quantization_config
        |
        v
InfiniLM safetensors 目录（6 shards）
        |
        v
GGUFBlockQuantization 按完整权重键解析类型和 shard
        |
        v
BaseLinear / Qwen3.5 模型层调用 linear_gguf
        |
        +-- decode：寄存器内 block decode + GEMV
        |
        +-- prefill：tile decode 到 workspace + GEMM
        v
BF16/F32 hidden -> 后续 attention、GDN、MLP、norm、lm_head
```

这条路线的关键原则是：量化格式信息从打包到运行时始终显式存在；若类型、shape 或映射不满足
约束，程序直接报错，而不是悄悄改走稠密权重。

## 5. GGUF 打包与权重映射

### 5.1 单一映射事实源

`/home/liuxd/InfiniLM/scripts/gguf_mapping.py` 是张量映射的单一事实源。每个映射条目描述：

- InfiniLM 参数名；
- GGUF tensor 名；
- 逻辑 shape；
- 是否保存为 blob；
- 支持的 GGML 类型；
- 融合张量的 slice 范围；
- 是否执行转置或 V head 置换；
- checkpoint 键和类型表键。

打包器、shape contract、内存预算和 C++ 运行时检查都基于同一映射计划，避免 Python 打包
规则和 C++ 加载规则分别维护后逐渐漂移。

### 5.2 支持的原生量化类型

| GGML 类型 | 类型 ID | 典型 block | 当前处理方式 |
|---|---:|---|---|
| Q8_0 | 8 | 32 个权重 / 34 B | 原生 U8 blob + GPU 解码 |
| Q4_K | 12 | K-quant block | 原生 U8 blob + GPU 解码 |
| Q5_K | 13 | K-quant block | 原生 U8 blob + GPU 解码 |
| Q6_K | 14 | 256 个权重 / 210 B | 原生 U8 blob + GPU 解码 |

例如：

- Q6_K，`K=5120` 时每行 `5120 / 256 * 210 = 4200 B`；
- Q6_K，`K=6144` 时每行 5040 B；
- Q6_K，`K=10240` 时每行 8400 B；
- Q6_K，`K=17408` 时每行 14280 B；
- Q8_0，`K=5120` 时每行 `5120 / 32 * 34 = 5440 B`。

因此 blob 的物理 shape 是 `[N, row_bytes]`，而不是普通线性权重的 `[N, K]`。运行时从
descriptor 中同时获得逻辑 K、GGML type 和 row bytes，并检查三者是否一致。

### 5.3 BF16 例外路径

当前模型中不属于四种原生类型的少量 IQ4_XS/IQ4_NL 权重在打包阶段显式解码为 BF16。
embedding 和 lm_head 在当前 v1 也采用 BF16 例外路径，以降低首次集成的复杂度。例外是映射
计划的一部分，不是运行时静默 fallback。

后续若补充 embedding gather-dequant 和量化 lm_head，可预计再节省约 2.51 GiB 权重显存。

### 5.4 融合权重、取向与置换

模型包含 GGUF 融合张量到多个 InfiniLM 参数的拆分。947 个产物条目多于 851 个实际消费
GGUF 权重，主要来自 48 个 GDN 层的融合 `attn_qkv` 一分为三。打包器按映射表定义的 slice
拆分，不能仅依靠名称替换。

对于线性注意力输出投影，还需要对 V head 执行 grouped-to-tiled 置换。当前实现按
`16 x 3 x 128` 的语义布局转换，使 GGUF 权重布局与 InfiniLM GDN 计算布局一致。

### 5.5 RMSNorm 约定修正

排查中发现 GGUF/RMSNorm gain 与原模型权重的表达约定不同。若把已经 baked `+1` 的 norm
权重再次加 1，会产生明显的逐层漂移。打包器新增 `_is_baked_plus1_norm()`，对所有 norm
权重统一判断，并排除不应套用该规则的张量。

### 5.6 打包器

主要脚本：

```text
/home/liuxd/InfiniLM/scripts/gguf_to_infinilm.py
/home/liuxd/InfiniLM/scripts/gguf_mapping.py
/home/liuxd/InfiniLM/scripts/gguf_transforms.py
```

打包器完成以下工作：

1. 读取 GGUF metadata 和 tensor directory；
2. 根据模型维度生成完整映射计划；
3. 对原生支持类型逐行复制 block bytes；
4. 对明确列出的例外解码为 BF16；
5. 执行 slice、转置、V permutation 和 norm convention 修正；
6. 写入带 index 的 safetensors shards；
7. 在 `config.json` 写入 `quantization_config` 和 947 项 `ggml_types`；
8. 复制 tokenizer/config 所需文件；
9. 对 shape、dtype、字节数和抽样原始 bytes 做自检。

脚本支持 `--dry-run` 和 `--skip-pack`，可以在不重复生成 23 GiB 产物的情况下审计映射或复用
已有分片。

### 5.7 最终产物组成

| 项目 | 数量/大小 |
|---|---:|
| safetensors 分片 | 6 |
| 总张量数 | 947 |
| 原生 U8 blob | 491 |
| BF16 张量 | 456 |
| U8 blob 体积 | 17.648 GiB |
| BF16 体积 | 5.615 GiB |
| 合计 | 23.264 GiB |

`ggml_types` 的类型直方图为：`dense_bf16=456`、`Q6_K=304`、`Q5_K=124`、
`Q8_0=59`、`Q4_K=4`。

## 6. InfiniLM 模型和量化框架接线

### 6.1 GGUFBlockQuantization

新增：

```text
/home/liuxd/InfiniLM/csrc/layers/quantization/gguf.hpp
/home/liuxd/InfiniLM/csrc/layers/quantization/gguf.cpp
```

`GGUFBlockQuantization` 的职责包括：

- 从模型配置读取 `ggml_types`；
- 按完整 checkpoint tensor key 查找具体 GGML 类型；
- 区分 `.weight_bytes` blob 和 BF16 `.weight`；
- 处理 fused shard 对应关系；
- 对需要的 shard 应用 V permutation 语义；
- 创建并调用 InfiniCore `linear_gguf_`；
- 对未知类型、缺失类型、shape 不符和不支持的 tensor parallel 显式报错。

### 6.2 Linear 层传递完整权重身份

普通量化框架只知道当前 Linear 的逻辑维度，但 GGUF 路由还必须知道它对应哪个 checkpoint
tensor。为此扩展了 `BaseLinear` 及相关线性层，使其保存 checkpoint stem 或 `shard_stems_`。
量化对象据此解析每个 fused shard 的类型，而不是按当前 C++ 对象名进行模糊匹配。

### 6.3 Qwen3.5/Qwen3.8 模型结构

完成了 Qwen3.5 风格模型在配置、注册、权重映射和运行时模块上的接入，包括：

- decoder layer；
- full attention；
- Gated DeltaNet/linear attention；
- MLP；
- GDN/SSM cache state；
- final norm 和 causal LM 输出；
- tokenizer/chat template 相关配置。

模型结构适配与 GGUF block 算子相互独立：前者决定“哪些张量放到哪里”，后者决定“某个
量化 Linear 怎样执行”。这种拆分是后续复用到其他架构的基础。

### 6.4 `ignore_eos` 语义修正

严格比较时发现，llama.cpp 的 `ignore_eos=true` 是在采样前屏蔽 EOS，而 InfiniLM 原有的
`stop_on_eos=false` 仅表示采到 EOS 后不停止，并不会阻止 EOS 被选中。这是采样语义差异，
不是算子误差。

为此扩展：

- C++ RankWorker Input 的 `suppressed_token_ids`；
- pybind 和 InferEngine 的字段传递；
- 低层 `GenerationConfig.ignore_eos`；
- 高层 SamplingParams 到每请求屏蔽列表的转换。

修正后 `ctx_03` 从第 27 token 分叉变为 32/32 完全一致，同时保留其他 stopping criteria。

## 7. 新增和扩展的算子

### 7.1 算子总表

| 算子/模块 | 类型 | 作用 | 默认状态 |
|---|---|---|---|
| `linear_gguf` | 新增 | 直接消费 GGUF U8 block 权重 | 启用 |
| GGML block decoders | 新增 | 解码 Q8_0/Q4_K/Q5_K/Q6_K | 启用 |
| register GGUF GEMV | 新增 | 小 M 的 decode/small-prefill | 启用 |
| tile dequant + GEMM | 新增 | 大 M prefill | 启用 |
| mixed add-RMSNorm | 扩展 | 承接实验性 F32 hidden 边界 | 普通路径不触发 |
| mixed GEMM fallback | 扩展 | BF16 权重乘 F32 hidden | 仅实验路径触发 |
| Q8A activation path | 实验新增 | Q8 激活量化后与 GGUF 权重计算 | 默认关闭 |
| F32 GGUF output | 实验扩展 | 指定 Linear 保留 F32 输出 | 默认关闭 |

### 7.2 `linear_gguf` 完整注册链

新增文件：

```text
/home/liuxd/InfiniCore/include/infiniop/ops/linear_gguf.h
/home/liuxd/InfiniCore/src/infiniop/ops/linear_gguf/linear_gguf.h
/home/liuxd/InfiniCore/src/infiniop/ops/linear_gguf/info.h
/home/liuxd/InfiniCore/src/infiniop/ops/linear_gguf/operator.cc
/home/liuxd/InfiniCore/src/infiniop/ops/linear_gguf/nvidia/linear_gguf_nvidia.cuh
/home/liuxd/InfiniCore/src/infiniop/ops/linear_gguf/nvidia/linear_gguf_nvidia.cu
/home/liuxd/InfiniCore/include/infinicore/ops/linear_gguf.hpp
/home/liuxd/InfiniCore/src/infinicore/ops/linear_gguf/linear_gguf.cc
/home/liuxd/InfiniCore/src/infinicore/ops/linear_gguf/linear_gguf_infiniop.cc
```

这条链覆盖 C API descriptor、InfiniCore C++ dispatcher、workspace 计算、设备 dispatch、
plan/run/cleanup。`info.h` 是 shape、dtype、GGML type 和 row bytes 契约的集中校验点。

### 7.3 GGML block 解码器

文件：

```text
/home/liuxd/InfiniCore/src/infiniop/ops/linear_gguf/ggml_blocks.h
```

这里实现 Q8_0、Q4_K、Q5_K、Q6_K 的共享 host/device block decoder。decoder 按 GGUF 的
原始位布局读取 scale、高位掩码和量化值。因为某些 row stride 只保证 2 字节对齐，不能假设
每个 block 都满足 4/16 字节对齐；实现使用安全的 byte load/拷贝方式，避免未对齐访问错误。

### 7.4 小 M 寄存器 GEMV

文件：

```text
/home/liuxd/InfiniCore/src/infiniop/ops/linear_gguf/nvidia/linear_gguf_gemv.cuh
```

执行方式：

1. 一个 warp 负责一个输出行；
2. warp lanes 沿 K 方向处理多个 GGUF block；
3. block 权重在寄存器中即时解码；
4. 与输入激活做 FP32 累加；
5. warp reduction 得到输出；
6. 正式路径把结果写为 BF16。

kernel 编译容量支持 `M<=16`。当前严格基线通过
`INFINI_GGUF_STRICT_SMALL_PREFILL_MAX_M=10` 选择 `M<=10` 使用该路径，因为它在当前
32x32 对拍中比更早切换到 prefill 路径更接近 llama.cpp 的归约结果。

### 7.5 大 M prefill

文件：

```text
/home/liuxd/InfiniCore/src/infiniop/ops/linear_gguf/nvidia/linear_gguf_dequant.cuh
```

当 M 超过小 M 路由阈值时，算子按 tile 把量化权重解码到临时 workspace，再调用 GEMM。
这种方式没有把整套模型权重永久还原为 BF16，只为当前 Linear 分配必要的临时 scratch，因而
仍符合 Route B。数值门覆盖到 `M=1024`，端到端验证覆盖 `M=12/64/512`。

### 7.6 mixed add-RMSNorm 扩展

修改文件：

```text
/home/liuxd/InfiniCore/src/infiniop/ops/add_rms_norm/info.h
/home/liuxd/InfiniCore/src/infiniop/ops/add_rms_norm/nvidia/add_rms_norm_nvidia.cu
```

为研究 BF16 物化边界，新增两类受限组合：

- F32 `a` + BF16 residual `b` + BF16 weight，FP32 求和和 RMS，输出 BF16；
- BF16 `a` + BF16 `b` + BF16 weight，FP32 求和和 RMS，输出 F32。

第一类用于让单个 F32 GGUF Linear 安全跨过 residual+norm 后回到 BF16；第二类用于 final
residual+RMSNorm 全 F32 实验。正常 BF16 模型路径保持不变。

### 7.7 GEMM mixed-dtype 修复与 fallback

修改文件：

```text
/home/liuxd/InfiniCore/src/infiniop/ops/gemm/nvidia/gemm_nvidia.cu
```

完成两项改动：

1. 修复 row-major 转置执行中交换 A/B 指针却没有同步交换 `a_type/b_type` 的问题；
2. 为 `BF16 large matrix x F32 hidden -> F32` 增加 batch=1 的 tiled register-GEMV fallback。

第二项是因为当前 cuBLAS 对该实际 mixed 组合返回 `CUBLAS_STATUS_NOT_SUPPORTED`。fallback
按 16 个 hidden column 分 tile，可覆盖任意 prompt M，但只在实验性 F32 final path 中使用。

### 7.8 Q8A 激活量化实验

`linear_gguf` 中还加入了受环境变量控制的 Q8A/Q8_1-like 激活量化路径，用于研究 llama.cpp
的激活量化和归约方式。它支持全 GGUF 类型或只命中某个 GGML type。实测该路径能修复个别
样例，但会使其他样例退化，因此保留代码用于研究，默认关闭。

## 8. 数值一致性优化

### 8.1 为什么两个后端不会天然完全一致

即使权重 block 解码公式正确，以下差异仍可能改变非常接近的 top-1：

- GEMV/GEMM 的分块和归约顺序；
- 中间结果何时从 FP32 舍入到 BF16；
- llama.cpp 的 Q8 激活量化与 InfiniLM 的 BF16 激活；
- RMSNorm residual sum 的物化 dtype；
- lm_head 累加精度；
- EOS 屏蔽等采样语义。

因此严格逐 token 一致性是独立的高标准验证项，不能简单等同于“算子正确性”。

### 8.2 当前接受的严格配置

当前接受配置包含：

- GGUF 四类型原生 block kernel；
- FP32 lm_head logits；
- 通用 `ignore_eos` 采样语义；
- small-prefill register path；
- `INFINI_GGUF_STRICT_SMALL_PREFILL=1`；
- `INFINI_GGUF_STRICT_SMALL_PREFILL_MAX_M=10`；
- Q8A、局部 F32 GGUF 输出和 final-FP32 全部关闭。

该配置得到：

```text
27 / 32 cases exact
920 / 1024 tokens match
```

剩余首分叉为：

```text
zh_04   @ token 28
zh_05   @ token 19
zh_06   @ token 4
code_04 @ token 4
math_04 @ token 1
```

### 8.3 已验证但未采用的实验

| 实验 | 结果 | 决策 |
|---|---|---|
| 全类型 Q8A | 能修复 `zh_05`，但五个重点例合计仅 75/160 token | 默认关闭 |
| Q6_K-only Q8A | 27/32、910/1024 | 退化，关闭 |
| Q5_K-only Q8A | 27/32、907/1024 | 退化，关闭 |
| layer0 attention out_proj F32 | `math_04` margin 明显恶化 | 关闭 |
| layer0 MLP down_proj F32 | `math_04` margin 明显恶化 | 关闭 |
| 强制 cuBLAS/寄存器 GEMV切换 | 未稳定修复剩余分叉 | 不采用 |
| final residual+RMSNorm F32 | 27/32、921/1024；修复 `zh_05` 但回归 `math_02` | 默认关闭 |

final-FP32 的全量结果比基线多匹配 1 个 token，但 exact case 仍为 27/32。它将 `zh_05`
修复为 32/32，同时使原本 exact 的 `math_02` 在 token 18 分叉。`math_02` 的参考 token 只落后
约 `5.95e-4`，说明这是非常临界的归约/舍入翻转，但在没有消除回归前不能作为默认优化。

## 9. 验证方法与结果

### 9.1 字节布局和解码公式

- 映射/字节布局审计：48 PASS / 0 FAIL；
- 四类型 block decode 交叉验证：47 PASS / 0 FAIL；
- 使用真实 GGUF blocks 做大规模抽样；
- 对 half 的 65536 种 bit pattern 做穷举扫描，并覆盖四种格式相关路径；
- blob 原始字节与源 GGUF 分类抽样逐字节一致。

### 9.2 Linear 数值门

- decode GEMV：两套独立产物各 56 PASS / 0 FAIL，cosine similarity 均大于 0.999；
- prefill：316 PASS / 0 FAIL；
- 覆盖四种 GGUF 类型、多种 N/K/M 和真实模型行字节；
- 端到端 prefill 覆盖 M=12、64、512。

### 9.3 映射、shape 与加载

- 映射计划共 947 条；
- 491 个 blob 的行字节均可整除且与 GGUF `n_bytes` 一致；
- 947 个产物 tensor 与引擎消费 tensor 双向集合一致；
- 947/947 shape 一致；
- 491 个 blob 在配置和 safetensors 中均为 U8；
- 6 个分片全量加载成功；
- 首个 blob forward 日志证明进入 `linear_gguf`，不存在稠密静默回退。

### 9.4 mini8 端到端

构造覆盖全部四种量化类型的 mini8 模型，61 个 blob 的分布为：

```text
Q6_K: 35
Q5_K: 12
Q8_0: 10
Q4_K: 4
```

模型成功执行 `generate()`，阶段检查 11 PASS / 0 FAIL，确认加载、路由、decode 和状态推进
形成闭环。

### 9.5 全量模型

全量 Qwen3.8-27B native-v2 已完成：

- 配置构造；
- 947 项权重装载；
- prompt prefill；
- autoregressive decode；
- GDN/SSM state 更新；
- 多 prompt 重复确定性；
- 32 x 32 严格 token 对拍。

llama.cpp 参考运行记录约为 prompt 29.0 token/s、generation 24.6 token/s。该数字只用于描述
参考后端，不是 InfiniLM 性能结论。InfiniLM 的正式吞吐、首 token 延迟、显存峰值和不同
prompt 长度曲线仍需要独立 benchmark 后才能下结论。

## 10. 最终结果与完成度判断

### 10.1 已完成

1. Qwen3.8-27B 模型架构可以在 InfiniLM 中构造和执行。
2. GGUF 到 InfiniLM 的映射、打包和配置生成已经完成。
3. Q8_0/Q4_K/Q5_K/Q6_K 四种原生权重算子已经完成。
4. decode 和 prefill 两条执行路径都已跑通。
5. 27B 全量模型可加载、生成，且明确走 U8 blob 原生 kernel。
6. shape、字节、数值、端到端和严格对拍均有报告留档。
7. 采样侧 `ignore_eos` 语义已与参考设置对齐。
8. 通用的 F32 边界、mixed add-RMSNorm 和 mixed GEMM 实验能力已经实现。

### 10.2 当前最终采用结果

```text
功能适配：成功
全量模型加载：成功
Prefill：成功
Decode：成功
原生量化类型：Q8_0 / Q4_K / Q5_K / Q6_K
严格一致性：27/32 cases，920/1024 tokens
严格目标 >=29/32：未完成，属于可选优化项
```

### 10.3 如何理解“成功”

如果验收标准是“让 InfiniLM 正确加载并推理 Qwen3.8-27B GGUF，并保留主要 GGUF 量化权重
不展开”，本工作已经完成。

如果验收标准额外要求“InfiniLM 与 llama.cpp 在固定 32 个样例中至少 29 个逐 token 完全
一致”，当前还差 2 个 exact case。后者是跨后端数值复现目标，不影响模型基本可用性；是否
继续投入，应由比赛规则、评测规则或业务需求决定。

## 11. 对其他 GGUF 模型的复用方式

### 11.1 可直接复用的通用部分

- safetensors U8 `weight_bytes` 存储约定；
- `config.json` 中的 `ggml_types` 和 `quantization_config`；
- Q8_0/Q4_K/Q5_K/Q6_K block decoder；
- `linear_gguf` C API、C++ API 和 NVIDIA backend；
- small-M register GEMV 与 large-M prefill dispatch；
- row bytes、dtype、shape 和无 silent fallback 契约；
- GGUF 原字节抽样、解码交叉验证和 Linear 数值门；
- mixed-dtype 边界诊断能力；
- 32x32 token 对拍、首分叉和 logits margin 工具。

### 11.2 每个新模型仍需适配的部分

- GGUF tensor name 到模型参数名的映射；
- fused tensor 的拆分/拼接规则；
- transpose、head permutation 或专家布局；
- norm gain 等模型特有权重约定；
- attention、MoE、SSM/GDN 等模型结构；
- cache/state 形状和生命周期；
- tokenizer、chat template、EOS 和 stop semantics；
- 模型实际包含但当前 kernel 未支持的 GGML 类型。

### 11.3 推荐的新模型适配流程

1. 读取 GGUF metadata，列出架构、tensor names、types、shapes 和 block bytes。
2. 新增模型维度类和 `build_plan()` 映射，不先写运行时特例。
3. 运行 dry-run，做源 GGUF 与目标模型参数的双向集合/shape 审计。
4. 将已支持的四种类型标记为 blob；其他类型明确列为 BF16 例外或新增 decoder。
5. 实现模型特有的 fused slice、transpose、permutation 和 norm convention。
6. 打包并执行全量字节/shape/dtype 自检。
7. 给各类型建立独立 block decode 和 Linear 数值门。
8. 用 mini 模型覆盖所有类型，完成 prefill+decode 闭环。
9. 加载全量模型，确认日志中首个和代表性权重进入 `linear_gguf`。
10. 最后做生成质量、确定性、性能和参考后端一致性测试。

这种流程下，新增一个结构相近且量化类型相同的模型，主要工作会集中在映射和模型结构层；
底层 GGUF 算子无需重复实现。

## 12. 当前限制与后续建议

### 12.1 当前限制

1. embedding 和 lm_head 尚未采用原生 GGUF kernel。
2. IQ4_XS/IQ4_NL 等 IQ 类型尚未原生支持。
3. GGUF blob 当前未实现 tensor parallel 切分。
4. prefill 已正确运行，但 tile 解码 workspace 和 GEMM 路由仍有性能优化空间。
5. 当前 32x32 与 llama.cpp 不是完全一致，剩余 5 个样例存在首分叉。
6. InfiniLM 正式性能数据尚未形成完整 benchmark 报告。
7. Q8A、F32 GGUF output、final-FP32 等研究路径均默认关闭。

### 12.2 优先级建议

如果目标是工程交付，建议按以下顺序继续：

1. 固化默认环境、构建说明和一键回归；
2. 测 InfiniLM 吞吐、TTFT、decode latency 和显存峰值；
3. 增加 native lm_head 和 embedding，降低约 2.51 GiB 权重占用；
4. 根据目标模型分布决定是否实现 IQ4；
5. 若有多卡需求，再设计 blob tensor parallel；
6. 只有评测明确要求时，再继续追求 `>=29/32` 的严格一致性。

若继续严格一致性，下一步应拆分 final-FP32 变量，分别测试：仅 F32 norm output、F32
residual sum + BF16 norm output、以及不同 lm_head reduction order。候选必须同时保留
`math_02` exact 并修复 `zh_05`，再允许跑完整 32x32，避免无方向地枚举局部精度开关。

## 13. 运行与复现要点

环境脚本：

```bash
source /home/liuxd/InfiniLM/scripts/gguf_routeb_env.sh
```

严格基线的关键环境变量：

```bash
export INFINI_GGUF_STRICT_SMALL_PREFILL=1
export INFINI_GGUF_STRICT_SMALL_PREFILL_MAX_M=10
```

实验变量默认不应设置：

```text
INFINI_GGUF_DECODE_Q8A
INFINI_GGUF_DECODE_Q8A_TYPE
INFINI_GGUF_F32_DECODE_OUT
INFINI_GGUF_F32_DECODE_OUT_MATCH
INFINILM_FINAL_NORM_FP32_FUSED
```

关键报告：

```text
/home/liuxd/tmp_routeb/reports/R3_strict_small_prefill_maxm10_32x32.json
/home/liuxd/tmp_routeb/reports/R3_compare_strict_small_prefill_maxm10_32x32.json
/home/liuxd/tmp_routeb/reports/R3_final_fp32_32x32.json
/home/liuxd/tmp_routeb/reports/R3_compare_final_fp32_32x32.json
```

构建时需要特别注意：只执行 `xmake build/install infiniop` 不足以保证 Python runtime 使用
最新库。运行时实际优先加载：

```text
/home/liuxd/InfiniCore/python/infinicore/lib/libinfiniop.so
```

因此 InfiniCore 改动后还必须执行 `xmake install _infinicore`，并核对安装目录与构建目录的
动态库哈希一致。此前多次“代码改了但结果不变”的根因就是只更新了 `/home/liuxd/.infini/lib`
而没有更新 Python 实际加载的副本。

## 14. 结论

本次工作已经建立了一条完整、可验证、可复用的 GGUF Route B：从 GGUF tensor 映射、原始
block bytes 打包，到 InfiniLM 类型路由、InfiniCore 原生 GPU 解码、prefill/decode，再到
全量 27B 模型生成和跨后端对拍，整个链路已经打通。

新增的核心能力不是只针对某一个 Qwen 权重文件的临时代码，而是一个可承载多模型的 GGUF
块量化线性算子框架。适配其他 GGUF 大模型时，可以复用存储协议、四类 block decoder、
`linear_gguf` 执行后端和验证体系，只需重点补充模型映射、结构接线和新的量化类型。

当前应将项目状态定义为：**GGUF 功能适配完成，主要量化权重原生执行成功；严格一致性达到
27/32，但 29/32 目标尚未完成且不是基本可用性的必要条件。**
