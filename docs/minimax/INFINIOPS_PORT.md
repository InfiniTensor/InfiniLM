# 迁移 `lightning_attention` 到 InfiniOps（新架构）

> 配套补丁：`docs/minimax/infiniops-lightning-attention.patch`（CPU + CUDA + pytest，6 文件 / +619 行）
> 旧架构补丁（归档基线用）：`docs/minimax/lightning-attention-infinicore.patch`

## 1. 为什么要迁到 InfiniOps

InfiniCore 已完成重构（`26f7382d refactor!: reduce InfiniCore to unified component architecture (#1406)`，2026-09-11）：顶层只剩 `.gitmodules` / `CONTRIBUTING.md` / `LICENSE` / `README.md` / `submodules`，算子实现归属 **InfiniOps**。

- 旧的 `infinicore::` C++ API（`op::*`、`nn::*`、`Tensor`、graph）在新版顶层已不存在。
- **InfiniLM 主干尚未迁移**（上游 `270feb3e`），因此现在无法把 minimax **模型**迁到新版；能且应该迁移的是**算子**。
- 本补丁交付 `lightning_attention_infinilm` 的 **CPU + NVIDIA CUDA** 后端与 pytest。

## 2. 补丁内容（6 个文件）

| 文件 | 说明 |
|---|---|
| `src/base/lightning_attention_infinilm.h` | 算子类（接口 + 元数据 + assert 校验），所有后端共享 |
| `src/native/cpu/ops/lightning_attention_infinilm/lightning_attention_infinilm.h` | CPU 参考实现（float32 累加，支持 f32/f16/bf16 与任意 stride） |
| `src/native/cuda/ops/lightning_attention_infinilm/kernel.cuh` | CUDA device kernel（每 `(batch, head)` 一块，每线程一列状态） |
| `src/native/cuda/ops/lightning_attention_infinilm/kernel.h` | CUDA launcher（`CudaLightningAttentionInfinilm<Backend>`，按 dtype/index dtype 分发） |
| `src/native/cuda/nvidia/ops/lightning_attention_infinilm/kernel.h` | NVIDIA vendor 绑定（`Operator<..., kNvidia>`） |
| `tests/test_lightning_attention_infinilm.py` | pytest：4 组形状 × 3 种 dtype，分别断言输出与状态池 |

## 3. 接口与语义设计

### 3.1 对齐目标与命名

按 `docs/operator-api-alignment.md` 的对齐顺序（PyTorch → vLLM → SGLang → ONNX → 库级公开封装 → CUDA/vendor → custom），lightning attention 在 PyTorch 中没有对应算子，最接近的公开实现是 Flash-Linear-Attention 的 `fused_recurrent_lightning_attn`。由于本算子带 InfiniLM 专有的"索引状态池"契约，按仓库既有 10+ 个先例（`paged_attention_infinilm`、`causal_softmax_infinilm`…）命名为 **`LightningAttentionInfinilm`**，注释中写明与 `dexp = exp(-slope)` 的对应关系。

### 3.2 参数顺序（InfiniOps 规范：输入 → 属性 → 输出）

```cpp
LightningAttentionInfinilm(q, k, v, slope,
                           initial_state, initial_state_indices, final_state_indices,
                           out)
```

### 3.3 递推语义

```
ratio[h] = exp(-slope[h])
S        = ratio[h] * S + outer(k_t[h], v_t[h])     # 先更新状态
out_t[h] = q_t[h] @ S                                # 再读状态（当前 token 权重为 1）
```

请求 `b` 从 `initial_state[initial_state_indices[b]]` 读状态，最终状态写回 `initial_state[final_state_indices[b]]`。

**关键契约：初始行不会被修改。** 读/写行相同时表现为就地更新；不同时等价于"先复制到目标行、再累加"。CPU 与 CUDA 实现都遵守，测试特意让读行 ≠ 写行来覆盖该契约（CUDA 侧通过在 kernel 开头把初始行协作拷贝到目标行、再在目标行上累加来保证）。

## 4. 服务器上需要下载/安装什么

### 4.1 基础依赖（必装）

| 依赖 | 版本/用途 | 安装方式 |
|---|---|---|
| Linux x86_64 | Ubuntu 22.04 等 | 租的机器自带 |
| **CMake** | ≥ 3.18（两个仓库都是 CMake 工程） | NGC 镜像通常自带，先 `cmake --version`；没有则 `pip install cmake` 或 `apt-get install -y cmake` |
| **C++ 编译器** | gcc-11+ / clang-16+（C++17） | `apt-get install -y build-essential` |
| **OpenMP** | CPU 后端 `find_package(OpenMP REQUIRED)` | `apt-get install -y libgomp1`（多半自带） |
| **Python** | ≥ 3.10 + pip | 镜像自带 |
| Python 包 | `torch`（测试参考实现）、`pytest`、`scikit-build-core` | `pip install torch pytest`；`.[dev]` 会带上构建依赖 |
| **InfiniRT** | InfiniOps 的前置依赖，**必须先装** | `git clone --recursive https://github.com/InfiniTensor/InfiniRT.git` + CMake 安装 |
| **InfiniOps** | 我们的补丁落在这里 | `git clone https://github.com/InfiniTensor/InfiniOps.git` |
| 网络 | configure 阶段访问 GitHub | CUDA 构建会 FetchContent 下载 CUTLASS（固定 commit + SHA256）；CPU 构建不需要 |

### 4.2 测 NVIDIA GPU 额外需要

| 依赖 | 说明 |
|---|---|
| **CUDA Toolkit（nvcc）** | ≥ 12；NGC PyTorch 镜像自带。注意 **InfiniRT 也要用 `-DWITH_NVIDIA=ON` 重新编译安装** |
| **CUTLASS** | 由 CMake `FetchContent` 自动下载，**不需要手动装**；网络受限时要预置或配代理 |
| **PyTorch（CUDA 版）** | 测试参考实现需要；NGC 镜像自带 |

### 4.3 完整命令（Linux 服务器）

```bash
# 0) 自检
cmake --version          # >= 3.18
g++ --version            # >= 11
python3 --version        # >= 3.10
nvcc --version           # 只有 GPU 机器需要

# 1) 编译安装 InfiniRT（GPU 机器同时打开 WITH_NVIDIA）
git clone --recursive https://github.com/InfiniTensor/InfiniRT.git
cmake -S InfiniRT -B build-rt \
      -DCMAKE_INSTALL_PREFIX=$HOME/infinirt \
      -DWITH_CPU=ON -DWITH_NVIDIA=ON
cmake --build build-rt -j
cmake --install build-rt

# 2) 取 InfiniOps 并应用补丁
git clone https://github.com/InfiniTensor/InfiniOps.git
cd InfiniOps
git apply /path/to/docs/minimax/infiniops-lightning-attention.patch

# 3) 构建安装 InfiniOps（CPU + NVIDIA）
python -m pip install ".[dev]" \
  --config-settings=cmake.define.INFINI_RT_ROOT=$HOME/infinirt \
  --config-settings=cmake.define.WITH_CPU=ON \
  --config-settings=cmake.define.WITH_NVIDIA=ON

# 4) 跑本算子的 pytest（device fixture 会自动覆盖 cpu 与 cuda）
pytest tests/test_lightning_attention_infinilm.py -v
```

- 只想先验 CPU：把第 1、3 步的 `-DWITH_NVIDIA=ON` 去掉即可（更快，也不需要 CUDA/CUTLASS）。
- 想只编本算子加速 configure/build：追加 `--config-settings=cmake.define.INFINI_OPS_OPS=lightning_attention_infinilm`。

## 5. 已知限制与下一步

- 本机（Windows，无 CMake）**未编译**本补丁；只做了 `py_compile`、人工复核与补丁反向校验（`git apply --check -R` 通过）。首次在服务器上编译可能有细节需要微调，按报错修即可。
- CUDA kernel 假定 `head_dim` 能放进一个 block（`head_dim <= Backend::max_block_size`，launcher 里有 assert），典型 MiniMax `head_dim = 128` 满足。
- **Ascend 后端**下一步：写到 `src/native/ascend/ops/lightning_attention_infinilm/`（InfiniOps 已有 AscendC 自定义 kernel 机制）。
- **InfiniLM 侧适配**：等上游完成 InfiniLM → 新栈迁移后，把 `MiniMaxLightningAttention` 的调用点换成 `infini::ops::LightningAttentionInfinilm`（一处调用 + 状态池形状对齐），模型其余代码不动。