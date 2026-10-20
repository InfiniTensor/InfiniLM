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
