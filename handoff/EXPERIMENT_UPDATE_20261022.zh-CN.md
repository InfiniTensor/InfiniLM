# 2026-10-22 实验与迁移补充

本页补充原双版本交接。旧版 fixed512 与当前重构版本的依赖和数学路径继续分别保留。测试的 InfiniLM 为 `5f91ad738583db8f5617aeae53b76523f85d1ec6`，Core/Ops/RT/CCL pins 沿用现有 5f91 清单。以下组件结果不能继承为模型吞吐。

## 投机实验的最新用户目标

只以每个用户实际得到的 decode tok/s 判优，排除 prefill。多 batch 仍逐请求报告；草稿生成、目标验证、选择、状态提交/回滚、重放及未隐藏通信全部计时。只数实际提交的新有效 token，不计拒绝提案、重放、EOS 和 EOS 后输出。普通三模型实验仍优化总吞吐/全部服务卡，投机为独立探索。

B1 优先，其次 B2/4/8/16；记录逐请求 TPOT 和非零块提交间隔，避免块内同 timestamp 形成虚假的零延迟。性能分母应为普通 `Engine.forward` AR；使用成对状态快照的诊断验证器不能作为普通 AR 速度基线。

## 8/16 窗口草稿与当前限制（本轮更新）

进一步查到 [Modal Kimi-K3 DFlash](https://huggingface.co/modal-labs/Kimi-K3-DFlash)，锁定 `c192d15a43407bf758b5ae0880d5c72052fef1de`。配置原有 `dflash_config.block_size=16`，不是将旧 Radix 草稿临时改为 16。官方普通 DFlash serving 路径丢弃 anchor 位置的输出，因此 W16 对应 K15 提案；RedHat DSpark 的 `sample_from_anchor=true` 则为 K8/W9，两者不能混用。

发布方 B1 的平均接受长度：Modal block8→16，GSM8K 5.905→7.984，HumanEval 6.013→8.358，MT-Bench 4.202→4.921，LongBench-v2 3.455→3.688。它们来自 NVIDIA B300/SGLang 的作者测试，并非 BW1000 实测，也不是与 RedHat/Radix 同提示、同采样的排名。Modal 权重 5,196,953,912 字节已在本地完整 SHA 校验：`0f0bd39d19c2dd0249de4f8216180f9729963402f902d50f19a156a8d358b733`；尚未在当前目标上验证接受率或在线速度。模型仓库未附 auto_map 指向的 Python 文件，不能直接假设 AutoModel 加载可用；适配须基于已审阅的 serving 实现。

[RedHat Kimi-K3 DSpark](https://huggingface.co/RedHatAI/Kimi-K3-speculator.dspark) 原生训练 8 提案，revision `38a88101e0d46bb22134b9da340f381b954d40d4`；9,489,807,826 字节权重也已本地完整 SHA 校验，正在传往集群，远端完整校验尚待完成。该包包含目标 embedding/head，不能把包总参数量全部视为额外草稿骨干。其 96Q/16KV、2048 causal SWA、taps `[24,48,72,88,92]`、MASK 163837 与旧 Radix 格式不同，禁止混用旧特征、mask 和非因果 attention。作者公开平均接受长度随任务约 3.1–6.4，不是承诺每轮接受 8 个提案。

判优以每位用户新提交的有用 token / 其 decode 时间为准。稳态近似为 `E[C]/E[Tcycle]`，相对普通 AR 为 `E[C]*T_AR/E[Tcycle]`；例如一轮有效 8 token、整轮耗时为普通单步 2 倍，对应 4 倍，而非 8 倍。草稿越长是否更好取决于真实前缀存活率和整轮成本。先 B1，再 B2/4/8/16、兼容 W8/W9/W16；普通 AR 使用 Engine.forward，同进程驻留权重 ABBA，prefill 排除，EOS/拒绝/重放不计输出。详见 [当前实验策略](evidence/speculative-user-decode-policy-r4218.json)。尚无合格在线投机加速。

## 已通过的草稿组件

完整 BF16 RadixArk Kimi-K3 DSpark，revision `3c5bac301d9cf392706189d82ed947feca6c2f0f`。原训练为 7 提案，8/16 是额外测试；16 是未训练长度外推，不能声称有效接受 16 token。输入为历史 3e0 目标的两个真实 first-block 特征 fixture，不是当前目标的在线接受率证据。

八卡各一次加载，B1/2/4/8 × K4/7/8/16：2,048 条普通对照、1,536 条阶段记录、4,096 条设备图对照，400 份原始 logits 经独立 CPU 有限值/argmax 检查。融合 Markov 算子的 256 组 CPU FP64 抽样通过；完整草稿 ID 与原 Torch 路径相同，正常退出和图/张量释放通过。

B1 草稿 host-completed 耗时，单位 ms：

| 提案数 | 原 Torch 固定前缀 eager | 原 Torch 固定前缀图重放 | 融合 Markov eager | 融合 Markov 图重放 |
|---|---:|---:|---:|---:|
| 4 | 13.78 | 9.61 | 11.25 | 7.21 |
| 7 | 16.84 | 12.56 | 12.47 | 8.37 |
| 8 | 17.73 | 13.31 | 12.74 | 8.52 |
| 16，外推 | 24.51 | 19.77 | 14.73 | 10.19 |

原 Torch 和融合算子各自是独立八卡实验，表中跨实验值为描述性对照，不是同一设备随机化的单项因果倍数。图内含 backbone、词表投影、Markov、confidence、argmax、结果拼接；图捕获与前缀初始化另记。变更 anchor/noise 验证设备输入可更新，固定前缀不变；尚未证明接受前缀推进、目标特征更新、变长、拒绝回滚或在线用户速度。代码、范围和精简证据见 [组件目录](experimental-latest-5f91ad7/speculative-draft-r4165/README.md) 与 [证据](evidence/speculative-decode-components-r4183.json)。

阶段计时显示 B1 backbone 约 8.3 ms，词表投影约 2 ms，Markov/confidence 在 K8/K16 时约 6.55/13.05 ms。融合实现以 wave64 复用 W2，合并 W1 gather、GEMV、BF16 bias 舍入和 base-logit 相加；未修改目标模型，未使用 F64 AllReduce。

## 目标验证器仍未修好

4148/4149/4150 完成实际 32 卡、一次完整权重加载、模式 6/7/8 对照及独立审计；正常 owner、QA、前后 AR 和追踪/未追踪重复通过，但三模式完整 128-token 继续生成均不同于冻结参考。

模式 6 首差异位置 33；只将多 token MLA attention 改为原生逐 query 调用后，模式 7/8 首差异位置均为 41，仍不能采用。22,860 份 CPU raw 诊断指出：首个 MLA 层的 attention/gated 输入已对齐，下一处为 MLA 输出投影，记录的 rank 0 有 4 个 BF16 元素不同。原始 logits 逐位相同只是诊断，失败依据仍是完整输出 token 不同。

只保存了 rank 0/6 的该投影输入，不能臆造其他六份。4173 最初按全八分片准备 fixture，在 GPU 提交前失败；保留失败。4175/4176 改为实际两个分片、每分片四个设备副本的本地 GEMM 回放：M2 差异在归约前存在，逐行 M1 精确复现串行参考。没有完整 TP8 归约或模型修复资格。

4179/4180/4181 已完成实际 32 卡测试与独立审计，模式 7/9/10 完整 token 均失败，首差异分别为 41/49/41；32 owner 正常关闭，前后 AR/QA/重复通过。它们没有在线性能资格。后续 22,860 份原始记录分析将模式 10 的更早剩余差异定位到局部 MLA KV 展开投影。4214/4215 用真实 layer7 输入、仅两个已保存分片 0/6、每分片四个硬件副本完成 8 卡回放，104 份 CPU 原始输出独立检查通过；分片 6 普通 M2 与逐行 M1 相差 2 个 BF16 元素，逐行 M1 恢复串行参考。此为本地投影诊断，没有假设其他六个分片、没有完整 TP8 归约或整模型修复资格；不能仅因 raw 差异采取模型结果，仍以最终完整 token 为门槛。

## GLM 两节点与 DeepSeek 长输出

GLM 16 卡 TP8/PP2，原层边界 `[0,38,78]`、640 context：四组构造/通信/只读缓存分类试验和独立审计通过。尾段最小空闲显存 B8 11.95 GiB，B16 9.07，B24 9.07，B32 3.33；B24 通过 8 GiB 加载门槛。逻辑缓存大小与实际分配不是同一口径，B16/B24 相同空闲量可能与分配器容量分档有关，不能从逻辑字节直接推断物理节省。

B32 尾段投影 K/V 各 3.125 GiB，latent 0.781、rotary 0.098、indexer 0.049 GiB；逻辑权重约 46.917 GiB。实际网络均为 IBext_v8，仍有 GDR Disabled 日志，未声称网络峰值。4167/4168 已完成原边界 B16/B24、M8/M16 整模型扫描并独立通过完整冻结 ID、QA、重复、16 owner/Slurm0/IB 验收；B16/M8、B24/M8、B16/M16、B24/M16 的 decode 中位分别 25.2770、28.2840、25.1255、27.2881 tok/s，含 prefill 分别 8.4914、9.4605、8.4581、9.3487。此为 512 输入/max128 的配置复测，不是代码因果收益。保留 8 GiB 加载及 6 GiB 运行余量。此前切点 40 为复用 indexer 层而被拒绝，不能重用其不合法方案。

另一个 B32 容量候选通过 persistent cache 精确分配将尾段空闲由 3.33 提到 7.55 GiB，结合 8 个通信通道达约 8.20 GiB并通过构造加载门槛。但其后实际 16 卡整模型 4199 在加载后 QA 发生真实 token 分歧，Slurm FAILED1、4200 队列拒绝，未达到正常全模型验收，不采用任何速度。尚未分离精确分配、通道数与数值路径的因果影响，不能认定内存容量证明等于模型可用。

DeepSeek-v4-Flash 最新 5f91、8 卡、512 输入/最多 512 输出、decode cohort16：B16/B32/B64 decode 中位数分别 **32.7598 / 35.0566 / 37.2190 tok/s**，含 prefill 分别 **24.7171 / 26.5720 / 27.9915**。各四次完整冻结 512-ID/EOS/重复/有限值/CPUargmax/正常 owner/Slurm0 与独立审计通过。B32/B64 是多个 16 行 cohort 顺序执行，原宽 N32/64 路径仍 token 失败。此为长度和配置复测，不能当成单项代码加速或在线混合流量成绩。

当前还没有合格的在线 Kimi 投机加速、三个模型的完整硬件极限、最优 TP/PP/EP 组合或 HTTP 混合流量结果，目标保持进行中。接手者先读私有最新进度检查点和实际 Slurm 句柄；观察超时不等于生产者终态，不可因此重复加载。
