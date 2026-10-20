# 海光 InfiniLM 双版本迁移交接

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
