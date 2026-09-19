# MiniMax 支持：归档与交接说明

> 本文记录「为 InfiniCore 重构做归档准备」这一轮（任务 A）实际做了什么、产出了什么、你还需要执行什么。
> 技术迁移细节见 `docs/minimax/PORTING.md`；算子补丁见 `docs/minimax/lightning-attention-infinicore.patch`。

## 1. 现状：重构已经落地

撰写本文时已核实上游状态（不是预告，是已发生）：

| 事实 | 证据 |
|---|---|
| InfiniCore 已完成重构 | `26f7382d refactor!: reduce InfiniCore to unified component architecture (#1406)`，提交时间 **2026-09-11 17:00 +0800** |
| 重构后顶层只剩集成件 | `26f7382d` 的顶层树仅含 `.gitmodules`、`CONTRIBUTING.md`、`LICENSE`、`README.md`、`submodules`；`include/`、`src/`、`xmake.lua`、CI workflow 全部删除 |
| 算子实现归属 InfiniOps | 该提交把 `submodules/InfiniOps` pin 在 `f890afb4b2327f13ccdd3c1b6b0d49567c5fe00d` |
| 我们落后 21 个提交 | 本地/归档基线 `35b46277`，`origin/main` 已是 `26f7382d` |
| **InfiniLM 上游尚未适配新版** | 上游 `InfiniTensor/InfiniLM` main = `270feb3e`，仅比我们的基线 `80bb09e` 多 5 个提交，且全是无关改动（readme、hygon、metax、多模态懒加载），没有任何 InfiniOps 适配 |

**由此得出的三条结论**

1. 我们写的 `lightning_attention`（位于 InfiniCore 顶层）**不会在重构后存活**，必须按 InfiniOps 规范重新落地。
2. 现在**整个 InfiniLM 主干都跑不了新 InfiniCore**（不只是 minimax），所以「适配新版」这件事目前**无法端到端验证**，性价比低、建议等上游跟进。
3. 我们的成果要能复现与评审，必须**锁定归档基线**：InfiniCore `35b46277` + 本补丁 ↔ InfiniLM `80bb09e` + MiniMax 改动。

## 2. 本轮实际执行的动作

### 2.1 状态巡检

对两个仓库做了完整的未提交状态巡检，把改动分成三类：

| 类别 | InfiniCore | InfiniLM |
|---|---|---|
| **本次 MiniMax 工作** | 4 个注册文件改动 + 6 个新增路径 | `python/infinilm/modeling_utils.py`（+52 行）、`csrc/models/minimax/`（10 个文件）、`test/models/minimax/`（3 个脚本） |
| **早先的 Windows 本地编译补丁**（与 MiniMax 无关） | `xmake.lua`（export_all） | `csrc/engine/distributed/tcp_rendezvous.cpp`、`csrc/models/kimi_k3/kimi_k3_pipeline_partition.hpp`、`csrc/models/minicpmv/minicpmv_model.cpp`、`xmake.lua` |
| **应排除的产物/垃圾** | `python/infinicore/bin/`、`python/infinicore/lib/` 下的 pyd/dll/lib | `python/infinilm/bin/`、`python/infinilm/lib/*.pyd|*.dll|*.lib`、根目录 `comp2011_1.cpp`（36 字节无关草稿） |

### 2.2 修复 `.gitignore`（两个仓库）

原规则只忽略 `*.so`（Linux 产物），Windows 构建产物会长期挂在 `git status` 里、随时可能被 `git add -A` 误提交。已在两个仓库补齐：

```
python/<pkg>/lib/*.pyd
python/<pkg>/lib/*.dll
python/<pkg>/lib/*.lib
python/<pkg>/lib/*.dylib
python/<pkg>/bin/
```

效果：两边构建产物已从 `git status` 消失，只剩真正的源码改动。

### 2.3 生成算子补丁

用 `git add -N`（intent-to-add）把新增文件纳入 `git diff`，再用 **git 自带的 `--output`** 写文件（避免 PowerShell 重定向写出 UTF-16/BOM 污染补丁），最后 `git reset -q` 撤销 intent-to-add，**索引与工作区均未被改动**。

补丁覆盖 **17 个文件 / +990 行 / 45,953 字节**：

```
include/infiniop.h                                    (注册)
include/infiniop/ops/lightning_attention.h            (C API)
include/infinicore/ops.hpp                            (注册)
include/infinicore/ops/lightning_attention.hpp        (C++ 包装)
python/infinicore/__init__.py                         (注册)
python/infinicore/ops/lightning_attention.py          (Python 包装)
src/infiniop/ops/lightning_attention/{info.h,lightning_attention.h,operator.cc}
src/infiniop/ops/lightning_attention/cpu/*            (CPU 参考实现)
src/infiniop/ops/lightning_attention/nvidia/*         (CUDA kernel + launcher)
src/infinicore/ops/lightning_attention/*.cc           (图/调度注册)
src/infinicore/pybind11/ops.hpp + ops/lightning_attention.hpp
```

**刻意不包含**：`xmake.lua`（早先的 Windows 本地补丁）、`.gitignore`（仓库卫生修复）——两者都不属于算子贡献，混入会污染将来的 InfiniOps 迁移评审。

校验方式：

- 在 InfiniCore 工作区对补丁执行**反向应用检查** `git apply --check -R`，退出码 0（说明补丁与工作区改动完全一致，因而对基线提交可正向应用）。
- 字节级检查：补丁仅含 LF（0 个 CR 字节、1134 个 LF），可直接在 Linux 服务器上 `git apply`。

### 2.4 CUDA kernel 静态复核（本机无法编译 CUDA，只能逐行审）

复核了 `nvidia/lightning_attention_nvidia.cu`、`nvidia_*.cuh`、`operator.cc`、`info.h`，并与仓库既有算子（`recurrent_gated_delta_rule`、`causal_softmax`）逐项对照。**发现并修复 6 处问题**，其中第 1 条是真实语义缺陷：

1. **【语义 bug】状态污染初始行**：原 kernel 让 `S` 直接指向状态池的 `init_row` 并就地累加，最后才把结果拷到 `final_row`。当 `init_row != final_row` 时，`init_row` 被污染；而 CPU 参考实现是把初始状态读进本地缓冲、只写回 `final_row` —— **两者语义不一致，且我们的算子单测正好覆盖该场景**（GPU 上会失败）。
   **修复**：先把 `init_row` 协作式拷贝到 `final_row`（配 `__syncthreads()`），再在 `final_row` 上就地累加。初始行保持只读，CPU/CUDA 语义一致，且不增加额外内存。
2. **【可移植性】改用 `INFINIOP_CUDA_KERNEL` 宏**：仓库约定用该宏（Hygon 构建下展开为 `__launch_bounds__(1024) __global__ void`），同时补上 `nvidia_kernel_common.cuh` 与 `<cstddef>` 头。
3. **【健壮性】`slope` 步长**：原 kernel 假设 `slope` 连续（`slope[h]`）。改为传入 `slope_stride` 按步长取值，不再依赖该假设（TP>1 时 `slope` 是 narrow 视图）。
4. **【潜在死锁】去掉 `if (tid >= D) return;`**：该早退分支位于 `__syncthreads()` 之前，一旦块大小不等于 `D` 就会死锁。改为在 `calculate` 中校验 `D <= maxThreadsPerBlock()`、固定按 `D` 个线程启动，并以注释写明该不变量。
5. **【编译阻塞项】signed/unsigned 比较**：`_opaque->internal->maxThreadsPerBlock()` 返回 `int`，而 `_info.D` / `_info.B` 是 `size_t`。写成 `size_t > int` 会在 InfiniCore 的 NVIDIA 构建下（`-Xcompiler=-Wall -Werror`）**直接编译失败**。已改为显式 `static_cast<size_t>(...)` 比较，并把 kernel 的 `slope_stride` 形参改成 `size_t`，避免设备侧混合符号运算。
6. **【输入校验】** `info.h` 增加索引张量 `stride(0) == 1` 的校验（CPU/CUDA 均按单位步长读索引），避免静默读错状态行；`calculate` 另加 `B <= 65535`（`gridDim.y` 上限）检查与 `(void)workspace` 显式消警。

另核实两点（非缺陷）：

- InfiniLM 的 `mamba_init/final_state_indices` 在各处理器中显式以 **int32** 构造（如 `qwen3_next_processor.py`：`infinicore.from_list(..., dtype=infinicore.int32)`），因此 CUDA 侧 i32-only 与真实调用路径一致；i64 仅 CPU 参考实现支持，CUDA 侧会明确返回 `BAD_TENSOR_DTYPE`。
- `PagedAttentionInfinilm` 等既有算子确认 `INFINIOP_CUDA_KERNEL`/`nvidia_kernel_common.cuh` 的用法一致，改动符合仓库现状。

### 2.5 修改后的回归验证（本机 CPU）

| 验证项 | 结果 |
|---|---|
| 算子 vs numpy 参考（decode `B=3,T=1`；prefill `B=2,T=6`） | 输出误差 9.54e-7 / 1.91e-6；状态误差 3.58e-7 / 2.38e-7 |
| 模型状态连续性（`prefill(5)` vs `prefill(4)+decode`） | 1.16e-10 |
| vs HF transformers `MiniMaxForCausalLM`（prefill / decode） | 5.59e-4 / 1.59e-4 |

三个测试全部通过，数值与修复前一致（说明修复未影响已验证的 CPU 语义）。

## 3. 产物清单

| 文件 | 说明 |
|---|---|
| `docs/minimax/lightning-attention-infinicore.patch` | 45,953 字节，17 文件 / +990 行，基线 InfiniCore `35b46277`，LF-only |
| `docs/minimax/PORTING.md` | 迁移到 InfiniOps 的对照与清单 |
| `docs/minimax/HANDOVER.md` | 本文件 |

## 4. 你需要执行的提交与推送

两个仓库分别操作。**路径列表刻意写全**，避免 `git add -A` 把垃圾文件或无关补丁一起提交。

### 4.1 InfiniCore

```bash
cd <你的 InfiniCore 工作区>
git checkout -b archive/minimax-lightning-attn

git add .gitignore \
        include/infiniop.h include/infinicore/ops.hpp \
        src/infinicore/pybind11/ops.hpp python/infinicore/__init__.py \
        include/infiniop/ops/lightning_attention.h \
        include/infinicore/ops/lightning_attention.hpp \
        src/infiniop/ops/lightning_attention \
        src/infinicore/ops/lightning_attention \
        src/infinicore/pybind11/ops/lightning_attention.hpp \
        python/infinicore/ops/lightning_attention.py

git commit -m "feat(infiniop): add lightning_attention op (CPU + NVIDIA) for MiniMax

Implements indexed-pool lightning attention (MiniMax-01 style):
  S <- ratio * S + k^T v ; o = q * S   with ratio[h] = exp(-slope[h])
CPU reference implementation plus a CUDA kernel, wired through the
infiniop C API, the infinicore C++ op layer and the Python bindings."

git tag minimax-lightning-attn
git push -u origin archive/minimax-lightning-attn
git push origin minimax-lightning-attn
```

`xmake.lua` 的本地 Windows 补丁**没有**包含在上面；如需保留本地构建能力，请单独提交（例如 `chore(build): export all symbols for MSVC builds`）。

### 4.2 InfiniLM

```bash
cd <你的 InfiniLM 工作区>
git checkout -b archive/minimax

git add .gitignore \
        python/infinilm/modeling_utils.py \
        csrc/models/minimax test/models/minimax docs/minimax

git commit -m "feat(minimax): support MiniMax-Text-01 (lightning attention + MoE)

- csrc/models/minimax: MiniMax model (hybrid lightning/softmax attention,
  block-sparse MoE, dense fallback), registered as minimax/minimax_m2
- python/infinilm/modeling_utils.py: _remap_minimax weight remapper
- test/models/minimax: op unit test, model smoke test, HF-aligned E2E test
- docs/minimax: porting guide and InfiniCore op patch"

git tag minimax-support
git push -u origin archive/minimax
git push origin minimax-support
```

`csrc/engine/distributed/tcp_rendezvous.cpp`、`csrc/models/kimi_k3/...`、`csrc/models/minicpmv/...`、`xmake.lua` 是早先的 Windows 本地补丁，建议另起提交，不要与 MiniMax 改动混在一起。

## 5. 服务器验证：必须锁版本

因为 `origin/main` 已经是重构版，服务器上**不能直接用 main**：

```bash
# InfiniCore：用重构前基线 + 我们的补丁（或直接 checkout 你的 archive 分支）
git clone --recursive https://github.com/<你的账号>/InfiniCore.git
cd InfiniCore
git checkout 35b46277bd666772c11bb417ad4231c5be492822
git apply /path/to/lightning-attention-infinicore.patch

# InfiniLM：checkout 你的 archive/minimax 分支（含 minimax 代码与 docs）
```

推荐节奏（省租机成本）：

1. **第一小时只做编译验证**：搭好环境 → 编译 InfiniCore（这一步就能暴露 CUDA kernel 是否可编译，是本机唯一无法验证的部分）。
2. 编译通过后再续租，跑：`test_lightning_attention_op.py` → `smoke_minimax.py` → `test_minimax_vs_hf.py`。
3. 若要测多专家 MoE，配 `num_experts=4, num_experts_per_tok=2` 之类的小配置（NVIDIA 上 `FusedMoE` 的 CUDA runner 可用）。

**CI 注意**：`.github/ci_config.yaml` 的 nvidia 镜像构建使用 `InfiniCore_BRANCH: __Branch_Name__`。若你的分支名在 InfiniCore 侧不存在，CI 会去拿上游 main（重构版）→ 必然编译失败。要么把 InfiniCore 的 archive 分支推到你的 fork 并使用同名分支，要么显式指定分支。

## 6. 三个需要特别注意的坑

1. **`comp2011_1.cpp`（InfiniLM 根目录）** 曾是一段 36 字节无关草稿（`for(int i =0; i<n; i++){ if }`），未跟踪。归档准备完成后该文件已从工作区移除（若你重新看到它，请勿提交；上面的 `git add` 列表也已刻意排除）。
2. **两个仓库必须成对使用**：InfiniLM 依赖 InfiniCore 的 `lightning_attention` 符号，只推一个会链接失败。归档请把两个 tag 配对记录（InfiniCore `35b46277` + 补丁 ↔ InfiniLM `80bb09e` + MiniMax 改动）。
3. **不要 `git pull` InfiniCore main**：会把重构版合进来、覆盖我们的算子目录结构。要跟进新版请在单独的 worktree / 克隆里做。

## 7. 下一步优先级

| 优先级 | 事项 | 说明 |
|---|---|---|
| P0 | 封存（第 4 节的提交 + tag + push） | 本地未提交改动是唯一副本，先落盘 |
| P1 | NVIDIA 服务器：编译验证（1 小时） | 验证 CUDA kernel 可编译；失败把日志给我，我来修 |
| P2 | NVIDIA 服务器：跑三个测试（半天） | 拿到 GPU 实测证据（评审材料里「只有 CPU 验证」是弱点） |
| P3 | 昇腾服务器：**只做平台不回归验证** | 当前 minimax 在昇腾跑不了（缺昇腾 kernel、MoE runner 为 CUDA 专属、CI ascend 段被注释） |
| P4 | InfiniOps 移植（可选加分） | 建议等 InfiniLM 上游跟进新版后再做，否则无法端到端验证；方案见 `PORTING.md` |



