# Agent 接手文档

## 最新用户验收要求（优先于下文历史 raw gate）

用户明确要求：**不需要所有数值逐位一致，保证最终输出的 token 一致即可。** 完整生成 token ID 序列（含 EOS）与同输入冻结参考一致是数值验收标准；logits 在重复行、rank、重复运行或精度变体间的逐位差异仅作诊断。仍验证有限值、CPU argmax 与实际所选 GPU token 一致、EOS、同配置重复、owner/设备/stream、正常退出和 Slurm 终态。raw SHA 用于文件完整性，不能据此要求模型数值逐位相同。

原生 BF16/FP32 为优先性能路径，F64 仅作参考和对照。F64 相对 BF16 的通信负载字节数为 4 倍，整模型慢多少必须实测。历史 raw-only 失败保留并标注旧规则；仅 raw 不同不再淘汰候选。完整 token ID 不同（包括空格 token）仍是失败，不做文本归一化放宽。

Kimi 已安排原生 BF16、FP32 两组完整模型试验，对冻结 F64 序列检查完整 token，并用相同 round-feedback 调度作 F64/候选/候选/F64 同进程对照。Deep 原生与匹配同步模式也按 token 标准重测。新精度路径尚无审计后的性能数字。具体代码见 `drivers/*r3560*`、`drivers/audit_kimi_token_r3562.py`、`drivers/*r3564*`，规则见 `evidence/token-acceptance-policy.json`。

## 任务与优先级

用户要求迁移三模型适配及优化工作，并交接历史高吞吐版和最新重构版。当前优化目标为**多 batch 下的有效总吞吐/全部服务卡数**，同时报告含 prefill 吞吐；不再以单用户 15 tok/s 为准入条件。旧版固定 512 数字必须在实际 EOS 场景下复测。完整 token 序列一致是最新数值验收标准；模型性能仍需有限值、EOS、正确计时、实际 owner 关闭和独立终态审计。

先读 `pins.json`、`PERFORMANCE.zh-CN.md`、`NUMERICS.zh-CN.md` 和 `evidence/recheck-status.json`。原集群任务与访问细节在用户本地私有附录，GitHub 不包含凭据。新服务器的分区、账号、DCU 类型、驱动、网络和模型路径须重新发现；不能继承旧集群 40 节点/320 卡配额或节点名单。

## 稳定与实验代码

| 模型 | 当前稳定构件 | 未完成或失败候选 |
| --- | --- | --- |
| GLM | 最新 Ops/Common `r2844/r2845`；Model 原始 `r2856`；多 batch host cache `r3072/r3073` | `r3323` fastpath 已审计通过 48.20 tok/s；待验证：`r3148` prefill cache、`r3348` 专家排序、长输出 G128/M8、`r3439` fastpath+G128/M32 |
| Kimi | 最新 Ops/Common `r2844/r2845`、Model `r2863` 的 B1 | `r3261/r3266` FP32 归约整模型失败；`r3393/r3395` F64 Common 已构建，等待完整模型验证 |
| Deep | 最新 Ops/Common `r2844/r2845`、Model `r2856` 的 B1 | `r2904` 多请求 FP32 归约仍失败；`r3443` 匹配同步控制与 attention/FFN/indexer F64 分级候选已安排隔离构建/组件测试 |

历史版本分别为 Kimi `r963`、GLM `r927`、Deep `r815`。它们依赖各自旧 SDK、旧 Common 和 BLAS policy，不能只复制一个 `.so` 到新版环境。

## 必须保持的工程约束

- 每个 Runtime、stream、张量与通信资源都由正确的 owner 线程操作。验证真实设备和 CPU affinity，不靠环境变量声明代替读取结果。
- 容器内 `/local` 必须绑定节点本地 tmpfs，Python cache 加 PID。共享 NFS 下的 import/pycache 曾造成长时间停顿；不能删除仍被活任务使用的缓存。
- 切换 SDK/Common/Model 使用新目录、新进程和加载映射 SHA，不能热替换运行中的库。generated public header 顺序不同曾导致检查失败，应验证语义与实际 ABI 后在隔离前缀处理。
- Kimi 私有 Model CommunicationGroup 与 Common 的布局不同。保留 `r3261` 修复：私有构造方法留在 Model，仅自由归约函数调用 Common，避免请求 Scope/TLS 分裂。
- 保留负例、失败运行与原始输出；旧 raw-only gate 已被用户的新 token 标准取代。只比较某一步 argmax 不足，必须验证完整序列；插桩结果仍须未追踪重放。
- 有效输出在第一个 EOS 结束，EOS 和结束后诊断输出不计入 useful tokens。decode、含 prefill、TTFT、HTTP 服务吞吐分开记录。
- Slurm `sacct`/`scontrol` 是作业状态依据。SSH、观察器或 launcher 超时不等于 GPU 任务失败，不能因此重复提交或取消仍在运行的任务。
- 不取消长期占用、不 reset GPU、不用无实际目的任务填满资源。旧集群有一张已隔离的故障卡，具体位置在私有附录；不能对该节点直接执行完整 TP8。

## 下一步执行顺序

1. 验证交接清单，恢复两套隔离源目录、模型和参考数据；在新节点完成单卡健康及小算子测试。
2. 分别构建 SDK、Common、GPU helpers 和 Model。记录编译器、ABI、动态库映射、设备与网络。先 1 节点，再 4 节点，不直接全量铺开。
3. 优先完成旧版 fixed512 与实际 EOS 复测，输出同一配置的两种口径。未经本轮完整审计不得称“已复现 116/63/25”。
4. Kimi 优先审计 r3560 原生 BF16/FP32 完整 token 与 F64 参考的一致性和精度 ABBA 吞吐。若 token 真正不同，保存第一处 token 差异，再缩小 partial/reduce/residual 边界；raw 差异只作诊断。
5. GLM 审计已在跑的排序、fastpath、长输出实验。两个旧 cache 矩阵曾在 21,000 秒 shell timeout 结束；不要无条件重跑整个长矩阵，拆成较短 ABBA。
6. Deep 在真实未追踪 B2 重放上分离局部计算与 collective。只有正确性通过再测多 batch 吞吐。
7. 在三个模型各自可用的配置上扫 G/M、副本和节点数；分母计入整个服务部署的全部卡，保留长短输出混合场景。多副本、HTTP、连续批处理和最优节点数尚未建立。

## 交接更新规则

每轮将源代码 SHA、实际终态、完整检查范围、性能口径和剩余问题写入新的 evidence 文件，随后更新 `recheck-status.json`。不要覆盖旧失败记录，也不要把仍在运行的记录标成成功。此交接不表示三模型优化目标已全部完成。

## 10 月 20 日晚追加状态（覆盖前文进行中条目）

1. 旧版 Kimi/GLM/Deep 固定 512 三组本轮复测已审计，性能见更新表；Kimi/GLM 实际 EOS 复测仍需独立审计。
2. 当前 `3e0` GLM 长输出 G128/M8 已通过，40.74–40.89 decode、30.84–30.91 含 prefill tok/s，32 卡。fastpath 长输出 M8、M32 与短输出 G128/M32 组合矩阵在测，不得先采用日志速度。
3. Kimi `r3419` 完整多 batch 失败，`r3486` 已捕获 B2 sort 缺空格 token220；保留原 literal gate 失败。B4/B8 独立完整验收 `r3501` 与 G32/M8 流水线 `r3483` 在测。先做完整 CPU raw/QA/关闭审计，再引用速度。
4. Deep B4 mode2 已通过 `r3494`，7.23–7.38 decode、6.71–6.85 含 prefill tok/s，8 卡。B2 raw 失败；B8 `r3498` 是入口输入约束失败，先修协议再重测。B16 要按 prefill micro8 分块，不能直接提交 8192 输入元素。
5. 新上游 `c3deba4`、Core `14a8bd5`、Ops `bcfe96d` 的 9 套源覆盖已完成，PadConstant→Pad 冲突显式解决并保留私有 guards。三 SDK 的 CPU-only build/load 最终 abort；读完整首错后隔离修复。Common/Model/helpers 须全部重建，禁止新旧混用。
6. 运行时归档 `r3432` 不含后续 Cast64/F64 修复库。`drivers/` 新增 Cast64 构建、Deep mode2 与新测试源码；从 `current` 源覆盖生成隔离候选后重建，不能把旧归档改名冒充修复版。

新服务器启动顺序与验收门槛仍按 MIGRATION。实际作业与节点映射留在本地私有附录；接手前读取最新 sacct，不能按本快照重提仍活跃作业。

更新至本轮末：新上游三个 SDK 已在健康节点通过 GPU 可见加载（每节点8卡小计算检查）；原 CPU-only 加载失败是无可见 ROCm 设备，保留负例。新 Common/Model、算子与吞吐仍须验收。旧版 Kimi 长文已独立审计 68.53–68.70/59.89–60.10 tok/s；当前 GLM G128/M32、最多128输出的组合优化已审计 60.51/20.83 tok/s（decode/含prefill），详见性能末表。Deep B8 分块 prefill 已重新测试；Kimi 流水线 producer 已结束，独立审计待完成。
