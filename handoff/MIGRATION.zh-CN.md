# 新服务器迁移步骤

## 最新用户验收要求（优先于下文历史 raw gate）

用户明确要求：**不需要所有数值逐位一致，保证最终输出的 token 一致即可。** 完整生成 token ID 序列（含 EOS）与同输入冻结参考一致是数值验收标准；logits 在重复行、rank、重复运行或精度变体间的逐位差异仅作诊断。仍验证有限值、CPU argmax 与实际所选 GPU token 一致、EOS、同配置重复、owner/设备/stream、正常退出和 Slurm 终态。raw SHA 用于文件完整性，不能据此要求模型数值逐位相同。

原生 BF16/FP32 为优先性能路径，F64 仅作参考和对照。F64 相对 BF16 的通信负载字节数为 4 倍，整模型慢多少必须实测。历史 raw-only 失败保留并标注旧规则；仅 raw 不同不再淘汰候选。完整 token ID 不同（包括空格 token）仍是失败，不做文本归一化放宽。

Kimi 已安排原生 BF16、FP32 两组完整模型试验，对冻结 F64 序列检查完整 token，并用相同 round-feedback 调度作 F64/候选/候选/F64 同进程对照。Deep 原生与匹配同步模式也按 token 标准重测。新精度路径尚无审计后的性能数字。具体代码见 `drivers/*r3560*`、`drivers/audit_kimi_token_r3562.py`、`drivers/*r3564*`，规则见 `evidence/token-acceptance-policy.json`。

## 1 准备两套隔离目录

建议为历史版和当前版分别使用 `runtime-legacy`、`runtime-current`，另设 `reference-assets`、`models` 和依赖源码缓存。不要让两套版本共用 `install`、`build`、Python native 扩展或写入中的结果目录。

先克隆临时交接分支并保存分支的实际 commit。`pins.json` 同时记录上游基线；交接分支包含覆盖层，只有 checkout 上游提交并不足以恢复实验代码。

```bash
git clone --branch handoff/hygon-inference-20261020 \
  https://github.com/InfiniTensor/InfiniLM.git InfiniLM-handoff
cd InfiniLM-handoff
python handoff/scripts/verify_package.py
python handoff/scripts/materialize.py --profile current \
  --destination /your/work/runtime-current --cache /your/work/upstream-current --fetch
python handoff/scripts/materialize.py --profile legacy \
  --destination /your/work/runtime-legacy --cache /your/work/upstream-legacy --fetch
```

`materialize.py` 只恢复源码布局，不编译，不下载权重，也不把旧实验标记为通过。缓存必须干净；已有目标目录会被拒绝。网络受限时，可在可联网机器准备同名的固定提交 Git 仓库，再去掉 `--fetch`。

## 2 迁移容器、模型与参考数据

原环境为 Ubuntu 22.04.5、DTK 26 系列、gfx936、Python 3.10，使用 Apptainer 1.3.4。具体软件包版本以 `evidence/environment.json` 为准。C++ ABI 由 `torch.compiled_with_cxx11_abi()` 读取，必须与全部 SDK、Common、Model、helpers 一致。

从原服务器安全复制已使用的 DTK sandbox 或原 SIF；其体积较大，不在 GitHub。新服务器还需兼容的宿主 DCU 驱动和 `/opt/hyhal`。仅安装通用 ROCm 不能假定得到同样环境。容器复制后做完整性检查，再验证实际设备、驱动和编译器。

权重单独迁移，保留 config、generation_config、tokenizer、词表、模板实现、safetensors index 及全部 shard。DeepSeek 的模型来源为 [Hygon ModelScope 模型](https://modelscope.cn/models/hygon/DeepSeek-V4-Flash-Channel-INT8-w8a8)。Kimi 和 GLM 使用原实验对应 checkpoint，不用同名新权重替代。小文件标识见 `reference-data-manifest.json`；权重 shard 的全文件校验需在复制后单独完成。

保留以下容器内路径，宿主目录由 `config.example.env` 配置：

| 容器路径 | 内容 |
| --- | --- |
| `/refactor` | 本次选择的独立 runtime 源码、安装、结果和控制目录 |
| `/old` | 冻结参考数据、旧离线 Python 依赖和 GLM 模型路径 |
| `/shared/Kimi-K3` | Kimi checkpoint |
| `/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8` | DeepSeek checkpoint |
| `/old/models/ZhipuAI-GLM-5.3` | GLM checkpoint |
| `/local` | 每个进程独立的节点本地 tmpfs scratch 和 Python cache |

解开单独交付的 `handoff-reference-data-*.tar.gz`，按其中相对 `results/`、`incoming/`、`control/` 路径放入对应 runtime。它保留历史 QA、失败样例和冻结参考，不含模型权重。`scripts/restore_references.py` 会校验文件和 archive SHA；动态实验输出名不应当作为输入文件预先创建。

另有分别约 49 MiB、44 MiB 的历史版和当前版运行时归档 `handoff-runtime-*-r3432.tar.gz`，包含原 SDK、Common、Model 和 helpers，可在相同容器栈上先恢复已使用的二进制。它们不含权重、容器和凭据，单独安全传输。先 materialize，再在空闲目标目录运行以下命令；脚本校验 archive、全部文件和符号链接，不覆盖不同的已有文件。恢复成功仍需要目标服务器的数值和性能验收。`r3428` 初版归档已由带符号链接元数据的 `r3432` 替代。

```bash
python handoff/scripts/restore_runtime.py /your/transfer/handoff-runtime-current-r3432.tar.gz \
  --profile current --runtime /your/work/runtime-current
```

## 3 发现新集群资源

```bash
sinfo -N -l
scontrol show partition
sacctmgr show assoc user="$USER" format=Account,Partition,QOS,GrpTRES,MaxTRES
```

根据新集群配置选择 account、partition、GRES、时限和节点。原账号的 40 节点/320 张名义卡和设备编号不能直接套用。先用一个空闲节点执行 preflight，读取每张卡的型号、架构、显存和实际小计算结果，再扩展到 4 个节点和通信检查。对错误卡按设备隔离；TP8 必须使用完整健康的八卡节点。

```bash
source /your/private/config.env
sbatch --nodes=1 --partition=YOUR_PARTITION --account=YOUR_ACCOUNT \
  --gres=dcu:8 handoff/scripts/job.sbatch
```

模板中的 `dcu:8` 和 CPU 数按新集群修改。preflight 是小任务，不会长期占用。网络应重新检查 IB/以太网、插件路径和实际带宽；先 Socket 基础可达，再验证可用 RDMA 插件。不要直接照搬旧集群 SHCA/UCX 设备名和动态库路径。

## 4 构建当前版本

在 **1 个构建节点** 内顺序执行，避免多个 Slurm rank 同时写同一 build/install 目录。需要 cmake、g++、xmake、Python 3.10、DTK nvcc、Torch、numpy、safetensors、transformers、pybind11 3.0.4，以及源码中固定的 json/spdlog。保留原始 GPU 浮点编译开关，特别是部分 attention/kernel 的 `--fmad=false`。

```bash
# config.env 中 TASK_ROOT 指向 runtime-current。
sbatch --nodes=1 handoff/scripts/job.sbatch \
  bash /handoff-repo/handoff/scripts/build_current.sh runtime
# runtime 成功后，再逐个模型构建；不要同时运行以下阶段。
sbatch --nodes=1 handoff/scripts/job.sbatch \
  bash /handoff-repo/handoff/scripts/build_current.sh sdk GLM
sbatch --nodes=1 handoff/scripts/job.sbatch \
  bash /handoff-repo/handoff/scripts/build_current.sh common GLM
sbatch --nodes=1 handoff/scripts/job.sbatch \
  bash /handoff-repo/handoff/scripts/build_current.sh model GLM
```

将 GLM 换成 Kimi 或 Deep，重复 SDK/Common/Model 三个阶段。脚本构建 fresh helpers，不从旧服务器复制 `.o` 以冒充重编译。新二进制 SHA 可以因环境变化而不同，需生成新的 build receipt，检查 `/proc/self/maps` 实际只加载一套预期 SDK 和 Runtime，再进行数值验收。不能把旧 receipt 的 hash 改成新 hash 并宣称原性能已经复现。

构建脚本是本次整理的迁移入口，尚未在新服务器上执行。原环境的详细编译命令、include、link、浮点开关在 `current/build-reference/` 和每个 `models/*/compile.sh`；诊断构建失败时以这些记录核对。新架构若不是 gfx936，需要移植内核并重新完成算子与模型验收。

## 5 恢复历史版本

历史版是完整源码快照加原始构建入口，SDK 源码位于 `dependencies/legacy/Ops/{Kimi,GLM,Deep}`；RT、CCL 也单独保存。实际使用的旧 SDK 指纹为 Kimi `6210897f…`、GLM `58735f67…`、Deep `8865b8d8…`。它们包含私有修改，不能仅根据上游 pin 重建一个干净 SDK 后当作同一版本。

`legacy/build-reference/` 的构建脚本含历史 receipt 和前置组件检查。迁移后的构建应重新生成自己的 provenance；它们不能在没有这些输入的空目录里直接作为一键安装器运行。`scripts/emit_legacy_compile.py` 可提取原始 Model 编译命令，配合脚本中的变量、独立 SDK/Common 和 GPU helper 构建；该工具不修改 benchmark 的数值检查。

```bash
python handoff/scripts/emit_legacy_compile.py Kimi --output /tmp/build-kimi-legacy.sh
```

Kimi 还需要 `kimi-grouped-fused-fp32-r21` helper；GLM 需要 routed、scale、pair、mask GPU 对象及 `glm_fp8_batched_r776`、`glm_fp8_cache_r807`；Deep 需要 routed、sinkhorn、gather、fused、post、scatter 和辅助 GLM 对象。相关原始内核及适配源码随交接保存。BLAS policy `r271` 也需从源码重建，不能漏掉 `LD_PRELOAD` 对实验路径的影响。

恢复完构建后，原样运行 `modern_kimi_token_only_r964.py`、`modern_glm_cache_view_r928.py`、`modern_deep_allselected_r816.py` 的冻结 QA 与重复检查，再运行实际 EOS 版本。先使用已有完整模板，不凭空换掉模型数学实现。对于缺失的旧参考文件，先恢复参考数据包或重建一套明确命名的新参考；不能删除 gate 继续计分。

## 6 正确性、性能与接续验收

按单卡小算子、8 卡 collective、完整 B1 QA、目标 batch QA/重复 raw、EOS、正常关闭的顺序验收。性能结果记录全部服务卡数，区分 decode、含 prefill 和 HTTP 服务。新旧版采用相同 prompt、G/M、输入输出上限、EOS 与节点，安排 ABBA；长矩阵拆成能在时限内完整结束的任务。

CPU 审计源码在 `drivers/audit_*.py`，它们对应明确的历史运行标签和路径。新运行必须使用新的 run ID、结果文件与独立审计入口，并保持检查语义，不覆盖原报告。迁移验收以新实验数据为准，不能只因 build/load 成功沿用表中的吞吐。
