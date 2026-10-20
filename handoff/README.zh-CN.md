# 海光 InfiniLM 双版本迁移交接

## 最新进展（2026-10-20T22:24:21.619999+00:00，覆盖后文旧状态）

最新5f91栈GLM的32卡TP8PP4、512输入/最多128输出、640上下文、96行容量完整扫描已结束；32owner正常关闭、Slurm0，3797独立审计后M8/M16全部QA及自然生成完整token与冻结3330相同且重复稳定。**M4 QA真实token不同，整套M4几何淘汰，不采用其速度**。B96/M16 decode中位64.8568 tok/s，含prefill20.5901，每全部32卡为2.0268/0.64344；prefill输入吞吐约120.23 input tok/s。是静态两类重复prompt、两次正反序测量，不是HTTP/混合流量，也不能和旧max512输出数值直接算回归或加速。详见 `evidence/qualified-GLM-profiles-r3797.json`。

GLM PP2旧试验在加载前尾段仅剩约3.1GiB，16上下文正常退出，3795独立审计；旧3773误按性能schema读取而失败，保留。idle trim实际释放0字节。新3783方案在每个Runtime内按head64/128、完整1M长度、theta8e6等完整键共享只读RoPE；弱池不留住RoPE/Runtime，组件3788全部8卡/双Runtime正常关闭。旧3783包装器漏CUDA include，在编译前失败，3788仅修包装器。模型数学、权重及原始1M RoPE表范围不变；本轮服务KV仍640。PP2同节点rank0构造后head实省5.875GiB、tail实省6GiB，tail空闲3.1992→9.1992GiB。加载前留8GiB（6GiB运行余量+2GiB加载事务），运行余量仍6GiB。3789调度同一3784 harness的PP2/PP4已加载并在QA；不能声称两节点性能已合格。3790排队独立验收。

Kimi3759修复后已通过8项QA与B1/B2等冻结AR，但M1/M2 B1投机K1仍同样在索引41出现6130对2762；拒绝重放token差异为0，整作业仍在更大batch，尚无投机加速可采用。需要同一前缀上首个接受块与串行的逐层定位；Common dense-prefill F32仅在rows>=512启动，本轮验证器N<=64，不能把它当已证实原因。3775把同池KDA接到实际Runtime并默认关闭，开关0/1、正确/错误ambient device共288组对照通过且3791审计；129状态槽算子每次可省48.375MiB，实际32行Model状态池33槽则每次12.375MiB，尚无整模型加速结果。

Kimi3761大B在加载前因128行同时预留516KV页，头段2.7–2.8GiB低于原3GiB门槛失败。不要放低门槛或当已跑通；若压缩KV页，须同时解耦当前KDA池num_blocks/4计算、保留129成对状态槽及物理page边界。DSpark在线特征/PP传输/增量draftKV仍未完成，投机排除主目标。

Deep最新B8已审计18.27–18.44 decode/15.43–15.56含prefill，8卡。3798在相同已验收3688模型/SDK/Common上新增B16整模型试验，512输入/最多512输出、micro4，逐行对重复prompt对应的冻结F64完整ID；3799审计已排队，不能继承旧B16结果。新服务器仍须重建各层依赖、正常owner/数值/网络/性能验收；本轮没有最优节点数、纯TP/EP、HTTP或硬件极限完成结论。

## 最新状态（2026-10-20T21:21:11.524499+00:00，覆盖后文历史进行中描述）

Kimi 投机首轮 B1/2/4/8/16 × K1/3/7 的15组均完整 token 不同且更慢，合格加速结果为0。投机独立于主目标。实际首次前向异常已定位 CausalConv1d；SDK-only Model3712越过该处后，原私有KDA调用又触发设备检查：新Runtime会恢复线程之前设备，raw helper不能假设 ambient CUDA 仍为owner。两个已确认报错的步骤及其PP等待进程已停止，父申请和健康作业未取消；不是正常owner关闭或性能验收。

3755隔离构建给raw KDA调用加显式owner DeviceGuard，保留原数值、设备与stream检查及调用者设备恢复。实际 Runtime 的正确/错误 ambient device 差分组件已通过（288 组），整模型仍待完整验收。 对应3759两组模式为KDA逐token舍入、KDA+MLA decode投影；B1–16、AR三次/投机两次。3761大batch B16/32/64要先等实际QA+冻结B1完整AR重复通过，3760/3762只在实际终态后独立审计。旧3736只作准备稿，不是已通过的性能版本。

GLM最新96/128行、640上下文的G64/96/128 × M4/8/16正在做正反序重复；3756全32 owner、完整ID/EOS/CPUargmax/库SHA/Slurm独立审计已排队，暂不采用未终态速度。Deep最新5f91的已审计结果保持8卡B8 decode18.27–18.44、含prefill15.43–15.56 tok/s，512输入/最多512输出。

DSpark仍只有离线组件，最新目标五层特征、PP传输、增量draft KV与在线接受率尚未接入。下一步先完成验证器完整token语义，再接训练草稿；不能把离线命中率当在线加速。KDA同池288组算子证明可省48.375MiB快照，但尚未用于Model。详细入口：`evidence/kimi-device-guard-r3765.json`、`drivers/*r3755*`、`drivers/*r3759*`、`drivers/*r3761*`。

## 最新用户验收要求（优先于下文历史 raw gate）

用户明确要求：**不需要所有数值逐位一致，保证最终输出的 token 一致即可。** 完整生成 token ID 序列（含 EOS）与同输入冻结参考一致是数值验收标准；logits 在重复行、rank、重复运行或精度变体间的逐位差异仅作诊断。仍验证有限值、CPU argmax 与实际所选 GPU token 一致、EOS、同配置重复、owner/设备/stream、正常退出和 Slurm 终态。raw SHA 用于文件完整性，不能据此要求模型数值逐位相同。

原生 BF16/FP32 为优先性能路径，F64 仅作参考和对照。F64 相对 BF16 的通信负载字节数为 4 倍，整模型慢多少必须实测。历史 raw-only 失败保留并标注旧规则；仅 raw 不同不再淘汰候选。完整 token ID 不同（包括空格 token）仍是失败，不做文本归一化放宽。

Kimi 已安排原生 BF16、FP32 两组完整模型试验，对冻结 F64 序列检查完整 token，并用相同 round-feedback 调度作 F64/候选/候选/F64 同进程对照。Deep 原生与匹配同步模式也按 token 标准重测。新精度路径尚无审计后的性能数字。具体代码见 `drivers/*r3560*`、`drivers/audit_kimi_token_r3562.py`、`drivers/*r3564*`，规则见 `evidence/token-acceptance-policy.json`。

交接日期为 2026 年 10 月 20 日。目标是在新一批服务器上恢复 GLM-5.3、Kimi-K3 和 DeepSeek-v4-Flash 的适配、正确性检查和吞吐优化。当前优化目标是**有效总吞吐除以服务实例全部卡数**；单请求 15 tok/s 已不作为当前实验准入条件。

本交接保留两套互相隔离的版本。不要把旧版的高吞吐数字直接贴到新版上，也不要把两套 SDK、Common、Model 动态库混合加载。

| 版本 | 用途 | 上层源代码基线 | 代码位置 |
| --- | --- | --- | --- |
| legacy-20261018 | 截图中固定 512 输入、固定 512 输出的历史吞吐复现 | `6b9cfa6b00cc7ca7431bc18e6c97936c4276e83e`，另含各模型私有修改 | `legacy/`、`dependencies/legacy/` |
| current-20261020 | 已锁定重构代码、实际 EOS 结束的吞吐和数值问题修复 | `3e0d89fe881ca9fb2411c6e35ffc913c7ea6654d`，另含各模型私有修改 | `current/`、`dependencies/current/`、`drivers/` |

源代码采用“固定上游提交 + 每个模型的修改覆盖层 + 原始适配器源码”的方式保存。仓库根目录仍是上游 InfiniLM；直接在根目录运行普通示例不会自动获得交接包里的全部优化。

## 阅读顺序

1. [性能与可复现性](PERFORMANCE.zh-CN.md)：先理解旧版固定输出和新版有效输出的区别。
2. [新服务器迁移步骤](MIGRATION.zh-CN.md)：配置硬件、模型、容器、构建与 Slurm。
3. [Agent 接手文档](AGENT_HANDOFF.zh-CN.md)：当前问题、证据、后续实验和验收要求。
4. [数据问题解决方案](NUMERICS.zh-CN.md)：Kimi、DeepSeek 的数值差异和验证顺序。
5. `pins.json`、`snapshot-manifest.json` 和 `evidence/`：版本、文件完整性和性能来源。

## 文件边界

GitHub 中保存可审查的 C++、GPU 内核、Python、构建入口、测试及审计源码。原始历史构建脚本保留了 `/refactor`、`/old`、`/shared` 这些**容器内路径约定**；迁移启动器通过绑定目录保留约定。

模型权重、私钥、账号、原集群接入地址、容器镜像和大体积原始日志不随代码发布。历史测试依赖的冻结参考数据按单独的数据包迁移；`reference-data-manifest.json` 列出文件和校验值。它们不是模型权重。原集群的作业、节点和访问方式放在本地私有附录中。

## 验证状态

旧集群上的已完成测试不等于新服务器已经验收。迁移后必须重新完成设备发现、源码构建、动态库加载检查、单请求 QA、对应 batch 的完整数值检查和重复性能测量。当前代码中的实验性 F64、专家排序、prefill 缓存等路径按各自状态启用，不能默认替换稳定基线。

## 追加交接（10 月 20 日晚）

旧版三模型固定 512 已按原有审计重新测出相近速度。新结果和限制见 [性能](PERFORMANCE.zh-CN.md) 与 `evidence/recheck-status.json`。当前性能基线仍为 `3e0d89fe`；新上游 `c3deba4` 的迁移单独记录于 [最新上游迁移](UPSTREAM_UPDATE.zh-CN.md)，尚无合格性能。

**先恢复已验证版本再继续修复。** `runtime-r3432` 只恢复原来稳定构件，不含后续 Cast64、Kimi F64 Common 或 Deep F64 Model；新增源码须在隔离目录重新构建和验收。

更新至本轮末：新上游三个 SDK 已在健康节点通过 GPU 可见加载（每节点8卡小计算检查）；原 CPU-only 加载失败是无可见 ROCm 设备，保留负例。新 Common/Model、算子与吞吐仍须验收。旧版 Kimi 长文已独立审计 68.53–68.70/59.89–60.10 tok/s；当前 GLM G128/M32、最多128输出的组合优化已审计 60.51/20.83 tok/s（decode/含prefill），详见性能末表。Deep B8 分块 prefill 已重新测试；Kimi 流水线 producer 已结束，独立审计待完成。

最新实际通信路径与 GLM RDMA 对照见 [网络检查](NETWORK.zh-CN.md)。本轮新增审计参考性能见性能末表；原生低精度新结果仍在测。


## 10月21日追加：加载、并行与独立投机实验

最新实际源码补充在 `experimental-latest-06d0c03/`，固定 LM06d0c03/Core046f012/Opsf998e4d；原 legacy/current 两份源快照逐字节保留。这是实验源码附加包，尚无最新06d整模型合格吞吐。新的已审计旧3e0性能、加载限制和Kimi投机试验见 [加载与并行](LOADING_AND_PARALLELISM.zh-CN.md)、[投机实验](SPECULATIVE_EXPERIMENT.zh-CN.md)。投机结果明确排除主目标指标。


## 最新补充：5f91重构依赖与投机修复实验

最新递归提交和实际源码见 [5f91实验接手](experimental-latest-5f91ad7/README.zh-CN.md)。RT/CCL和三模型所有层已重建，新32卡高速通信组件验收通过。06d GLM与Deep已有独立全token验收性能；5f91整模型仍在测，不能混标。Kimi首轮B1–16的15组投机比较均发生真实token差异，未得到可采用的投机加速；KDA/MLA修复两组及B16/32/64流水线第三组继续独立测试，全部排除主指标。


## 本轮终态补充（覆盖前文“在测”）

最新5f91栈Deep-v4-Flash已完成独立审计：8卡B8、512输入/最多512输出、attention BF16->FP32 SUM->BF16，decode18.27–18.44 tok/s，含prefill15.43–15.56 tok/s，完整token与冻结F64参考一致，8个owner正常关闭、Slurm0。不使用F64 AllReduce；四个长例都到512上限，不代表混合真实流量。见 `evidence/qualified-latest-Deep-r3729.json`。

Kimi M1/M2最新两组均在首次前向失败：operator has no active implementation；尚未进入token与性能验收。普通IndexSelect8卡小测试通过；所测include顺序及新旧Embedding注册均相同，不能认定header改动3712已修复根因。已准备一次带过滤异常调用栈的4节点原失败重放3731，复用stage0部分CPU缓存，先定位实际算子再重新加载多组。

大batch流水线3703在加载前因128行上下文仍套用了32行尺寸断言而失败，32owner正常关闭，未取消父任务或任何健康步骤；另外发现继承的大写FROZEN环境变量名需一致。B32/64还须把paired-state shadow偏移由17改65，并验证128行/516页/129状态槽。该流水线没有可用速度，不能按已运行成功交接。旧M1/M2和原始15组token负例均保留。

GLM cap640两组3698因retrieval QA的518输入+固定256输出上限检查失败，属于驱动容量约束；3721把QA上限设为min(256,640-input+1)，仍要求完整EOS和token参考，未放宽质量或6GiB余量。96/128行两组继续验证；没有合格大batch新结果可采用。

KDA129槽disjoint read/write同池测试3733用于判断能否省去每层整池快照，单次约48.4MiB；3728仅因缺动态helper路径在kernel前失败，不能当算子失败。独立算子通过也不代表整模型已修复。训练DSpark在线集成仍未完成，当前没有可采用的投机加速，投机始终排除主目标指标。
