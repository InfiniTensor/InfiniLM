# 迁移 `lightning_attention` 到 InfiniOps

本文件说明如何把 MiniMax 所需的 `lightning_attention` 算子，从**重构前**的 InfiniCore 形态迁移到**重构后**的 `submodules/InfiniOps`。

- 配套补丁：`docs/minimax/lightning-attention-infinicore.patch`
- 补丁基线：InfiniCore `35b46277bd666772c11bb417ad4231c5be492822`
- 补丁规模：17 个文件 / +990 行（含 CPU 与 NVIDIA 两套实现）

## 1. 背景（重构已落地）

InfiniCore 的重构已经完成并合入上游：

- 提交：`26f7382d refactor!: reduce InfiniCore to unified component architecture (#1406)`（2026-09-11 17:00 +0800）
- 重构后 InfiniCore 顶层只剩 `.gitmodules` / `CONTRIBUTING.md` / `LICENSE` / `README.md` / `submodules`；`include/`、`src/`、`xmake.lua`、CI workflow 全部移除。
- 算子实现归属 `submodules/InfiniOps`（该提交 pin 在 `f890afb4b2327f13ccdd3c1b6b0d49567c5fe00d`），要求**接口与主流开源框架一致**、合并要求严格。
- `InfiniLM` 上游（`270feb3e`）**尚未适配**新版 InfiniCore，因此当前阶段单独移植 minimax 也无法端到端验证。

因此当前位于 InfiniCore 顶层的算子实现（`src/infiniop/ops/lightning_attention/` 等）**不会在重构后保留**，需要按 InfiniOps 的规范重新落地。数学推导、CPU 参考实现与 CUDA kernel 主体可以直接复用；需要重写的是接口层与构建/注册部分。

## 2. 算子契约（语义不随迁移改变）

### 2.1 数学语义

MiniMax-01 风格的 Lightning Attention（带 ALiBi 式逐头衰减），逐 token 递归：

```
ratio[h] = exp(-slope[h])
S        = ratio[h] * S + outer(k_t[h], v_t[h])     # 先更新状态
o_t[h]   = q_t[h] @ S                                # 再读状态（当前 token 权重为 1）
```

即 `S ← ratio ∘ S + kᵀv`，`o = q·S`。注意状态**先更新后读取**，当前 token 对自己的衰减为 0（权重 1），这一点与实现无关，是模型语义的一部分。

### 2.2 张量约定（迁移时保持不变）

| 张量 | 形状 | 说明 |
|---|---|---|
| `out` | `[B, T, H, D]` | 输出，末维连续 |
| `initial_state` | `[pool_size, H, D, D]` | 状态池，F32 累积 |
| `q` / `k` / `v` | `[B, T, H, D]` | 末维连续 |
| `slope` | `[H]` | fp32，逐头衰减系数 |
| `initial_state_indices` | `[B]` | int32/int64，读入状态的行号 |
| `final_state_indices` | `[B]` | int32/int64，写回状态的行号（原位写回池） |

### 2.3 索引池（indexed-pool）语义

每个请求 `b`：

1. 从 `initial_state[initial_state_indices[b]]` 读取初始状态；
2. 在 `[B, T, H, D]` 上按 token 递归（`T=1` 即 decode）；
3. 把最终状态**原位写回** `initial_state[final_state_indices[b]]`。

该语义与 `recurrent_gated_delta_rule` / `chunk_gated_delta_rule` 的 indexed pool 一致，便于 InfiniLM 复用 `mamba_init_state_indices` / `mamba_final_state_indices` 调度机制。

## 3. 当前实现（重构前形态）

| 文件 | 作用 |
|---|---|
| `include/infiniop/ops/lightning_attention.h` | C API：`infiniopCreate/GetWorkspace/Destroy/LightningAttention` |
| `src/infiniop/ops/lightning_attention/info.h` | 描述符校验（形状、dtype、末维连续、索引范围） |
| `src/infiniop/ops/lightning_attention/lightning_attention.h` | `DESCRIPTOR(NAMESPACE)` 宏 |
| `src/infiniop/ops/lightning_attention/operator.cc` | 按设备分发（CPU / NVIDIA） |
| `src/infiniop/ops/lightning_attention/cpu/*` | CPU 参考实现（F32/F16/BF16，支持任意 strides） |
| `src/infiniop/ops/lightning_attention/nvidia/*` | CUDA kernel（**目前仅 F32**）+ launcher |
| `include/infinicore/ops/lightning_attention.hpp` | C++ 包装 `infinicore::op::lightning_attention_` |
| `src/infinicore/ops/lightning_attention/*.cc` | 图/调度注册（`INFINIOP_CACHABLE_DESCRIPTOR`） |
| `src/infinicore/pybind11/ops/lightning_attention.hpp` | Python 绑定 |
| `python/infinicore/ops/lightning_attention.py` | Python 包装 |

注册点（重构后由代码生成替代）：

- `include/infiniop.h` 增加一行 `#include "infiniop/ops/lightning_attention.h"`
- `include/infinicore/ops.hpp` 增加 `#include "ops/lightning_attention.hpp"`
- `src/infinicore/pybind11/ops.hpp` 增加 include 与 `bind_lightning_attention(m);`
- `python/infinicore/__init__.py` 增加 `from infinicore.ops.lightning_attention import lightning_attention`

## 4. 目标形态（InfiniOps）与对照

InfiniOps 的算子是一个继承 `Operator<T>` 的类，kernel 与 launcher 分离：

```
src/base/<op>.h                                  # 算子类（参数校验 + 元数据）
src/native/<backend>/[<vendor>/]ops/<op>/kernel.h        # launcher
src/native/<backend>/[<vendor>/]ops/<op>/kernel.cuh      # device kernel
```

| 当前实现 | InfiniOps 目标 | 需要改什么 |
|---|---|---|
| `include/infiniop/ops/lightning_attention.h`（C API + descriptor） | `src/base/lightning_attention.h`（`class LightningAttention : public Operator<LightningAttention>`） | **删除 descriptor/创建销毁接口**，改为构造函数做校验 + 记录 strides/元数据 |
| `info.h` 的 `utils::Result` + `CHECK_DTYPE` 校验 | 构造函数中的 `assert` | InfiniOps **禁用异常**；错误信息需含 `__FILE__`/`__LINE__`/`__func__` |
| `operator.cc` 的设备分发 | 由 InfiniOps 的分发机制按 backend 选择实现 | 删除手写分发 |
| `cpu/*.cc` | `src/native/cpu/ops/lightning_attention/` | 复用算法，改为 launcher 形态 |
| `nvidia/*.cu` + `*.cuh` | `src/native/cuda/ops/lightning_attention/{kernel.h, kernel.cuh}` | 复用 kernel 主体；**必须补 fp16/bf16** |
| `include/infinicore/ops.hpp`、pybind、Python 包装 | 生成的 `operator_call_instantiations.h` + `GENERATE_PYTHON_BINDINGS` | 删除手写注册与绑定 |

参数顺序也要改成 InfiniOps 规范：**输入在前 → 属性居中 → 输出最后**，例如：

```cpp
LightningAttention(const Tensor q, const Tensor k, const Tensor v,
                   const Tensor slope, const Tensor initial_state,
                   const Tensor initial_state_indices,
                   const Tensor final_state_indices, Tensor out);
```

## 5. 可以直接复用 / 必须重写

**可直接复用**

- 递推公式与代数推导（`S ← ratio ∘ S + kᵀv`，`o = q·S`）。
- CUDA kernel 主体：按 `(batch, head)` 分块的逐列更新逻辑，共享内存缓存 `k`/`q` 行的做法。
- CPU 参考实现的循环结构与 stride 处理。
- 边界校验项（末维连续、`D <= 1024`、dtype 一致性、索引 dtype）。

**必须重写 / 补齐**

1. 接口层（类 + assert 校验，见第 4 节）。
2. **dtype 覆盖**：InfiniOps 的算子普遍只接受 fp16/bf16（例如 `PagedAttentionInfinilm` 明确 `assert` 仅 f16/bf16），当前 kernel 只有 fp32，这是能否被合入的硬门槛。
3. kernel 文件命名与拆分规范（`kernel.h` + `kernel.cuh`，非模板 kernel 也要求头/源分离）。
4. 代码风格：Google C++ Style + 仓库自带 `.clang-format`；注释与错误信息用英文完整句。
5. 构建：改用 InfiniOps 的 CMake（`WITH_NVIDIA` / `WITH_ASCEND` / … 选项），不再依赖 InfiniCore 的 xmake glob。

## 6. 接口命名建议（对齐开源）

InfiniOps 要求接口与主流开源框架一致，命名优先级为 **PyTorch → ONNX → CUDA API**。

- Lightning Attention 的开源等价物是 Flash-Linear-Attention（fla）的 `chunk_lightning_attn` / `fused_recurrent_lightning_attn`；vLLM 的 MiniMax 实现也使用同名 kernel。
- 因此建议参数与语义对齐 fla：`q, k, v, slope, initial_state (+ indices), out`，并在文档中标注「decode 走 recurrent、prefill 走 chunk」。
- 若维护者认为暂无可对齐的开源算子，可沿用仓库既有先例：`PagedAttentionInfinilm` 采用 `xxx_infinilm` 后缀并标注
  `[[deprecated("Migrate to an open-source-aligned operator when available.")]]`；此时命名为 `LightningAttentionInfinilm` 更符合现状约定。

## 7. 昇腾（Ascend）实现位置

昇腾实现**不要**写到 InfiniCore 的 `src/infiniop/ops/*/ascend/`（同样会被清理），而应写到：

```
src/native/ascend/ops/lightning_attention/
```

InfiniOps 已有 `src/native/ascend/custom/`（AscendC 自定义 kernel）与 CMake 开关 `BUILD_ASCEND_CUSTOM`，可直接复用该机制。InfiniLM 侧 CI 目前 `ascend:` 段是注释状态，需要一并启用。

## 8. 迁移后自检清单

1. 数值：与补丁中的 CPU 参考实现逐值比对（decode `T=1` 与 prefill `T>1` 两种场景，含「初始行 ≠ 写回行」的索引池场景）。
2. 状态连续性：`prefill(N)` 与 `prefill(N-1)+decode(1)` 的末位 logits 一致（当前 CPU 实测差异 1e-10 量级）。
3. 端到端：对照 HF transformers `MiniMaxForCausalLM` 的 prefill/decode logits（当前实测 5.6e-4 / 1.6e-4）。
4. dtype：至少覆盖 fp16、bf16（数值比对可用 CPU 参考或 torch 参考）。
5. 形状/边界：`D ∈ {64, 128}`、非连续输入、`B` 多请求、索引池行冲突（多个请求读写同一行）等。


