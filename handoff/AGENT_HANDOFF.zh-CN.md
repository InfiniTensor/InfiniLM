# Agent 接手文档

## 任务与优先级

用户要求迁移三模型适配及优化工作，并交接历史高吞吐版和最新重构版。当前优化目标为**多 batch 下的有效总吞吐/全部服务卡数**，同时报告含 prefill 吞吐；不再以单用户 15 tok/s 为准入条件。旧版固定 512 数字必须在实际 EOS 场景下复测。不要把模型适配完成、算子通过、任务提交、日志中出现速度或 token 相同视为完整模型验收。

先读 `pins.json`、`PERFORMANCE.zh-CN.md`、`NUMERICS.zh-CN.md` 和 `evidence/recheck-status.json`。原集群任务与访问细节在用户本地私有附录，GitHub 不包含凭据。新服务器的分区、账号、DCU 类型、驱动、网络和模型路径须重新发现；不能继承旧集群 40 节点/320 卡配额或节点名单。

## 稳定与实验代码

| 模型 | 当前稳定构件 | 未完成或失败候选 |
| --- | --- | --- |
| GLM | 最新 Ops/Common `r2844/r2845`；Model 原始 `r2856`；多 batch host cache `r3072/r3073` | `r3148` prefill cache；`r3348` 专家排序；`r3323` fastpath；长输出 G128/M8 |
| Kimi | 最新 Ops/Common `r2844/r2845`、Model `r2863` 的 B1 | `r3261/r3266` FP32 归约整模型失败；`r3393/r3395` F64 Common 已构建，等待完整模型验证 |
| Deep | 最新 Ops/Common `r2844/r2845`、Model `r2856` 的 B1 | `r2904` 多请求 FP32 归约仍失败；tracing 扰动原失败窗口 |

历史版本分别为 Kimi `r963`、GLM `r927`、Deep `r815`。它们依赖各自旧 SDK、旧 Common 和 BLAS policy，不能只复制一个 `.so` 到新版环境。

## 必须保持的工程约束

- 每个 Runtime、stream、张量与通信资源都由正确的 owner 线程操作。验证真实设备和 CPU affinity，不靠环境变量声明代替读取结果。
- 容器内 `/local` 必须绑定节点本地 tmpfs，Python cache 加 PID。共享 NFS 下的 import/pycache 曾造成长时间停顿；不能删除仍被活任务使用的缓存。
- 切换 SDK/Common/Model 使用新目录、新进程和加载映射 SHA，不能热替换运行中的库。generated public header 顺序不同曾导致检查失败，应验证语义与实际 ABI 后在隔离前缀处理。
- Kimi 私有 Model CommunicationGroup 与 Common 的布局不同。保留 `r3261` 修复：私有构造方法留在 Model，仅自由归约函数调用 Common，避免请求 Scope/TLS 分裂。
- 保留负例、失败运行与原始输出；数值误差变小、重复行 argmax 一致、插桩后通过，都不能代替原检查。
- 有效输出在第一个 EOS 结束，EOS 和结束后诊断输出不计入 useful tokens。decode、含 prefill、TTFT、HTTP 服务吞吐分开记录。
- Slurm `sacct`/`scontrol` 是作业状态依据。SSH、观察器或 launcher 超时不等于 GPU 任务失败，不能因此重复提交或取消仍在运行的任务。
- 不取消长期占用、不 reset GPU、不用无实际目的任务填满资源。旧集群有一张已隔离的故障卡，具体位置在私有附录；不能对该节点直接执行完整 TP8。

## 下一步执行顺序

1. 验证交接清单，恢复两套隔离源目录、模型和参考数据；在新节点完成单卡健康及小算子测试。
2. 分别构建 SDK、Common、GPU helpers 和 Model。记录编译器、ABI、动态库映射、设备与网络。先 1 节点，再 4 节点，不直接全量铺开。
3. 优先完成旧版 fixed512 与实际 EOS 复测，输出同一配置的两种口径。未经本轮完整审计不得称“已复现 116/63/25”。
4. Kimi 按 `NUMERICS.zh-CN.md` 的 F64 路径完成审计和整模型最早失败窗口测试。若失败，保留故障窗口并缩小 partial/reduce/residual 边界；不要放宽 gate。
5. GLM 审计已在跑的排序、fastpath、长输出实验。两个旧 cache 矩阵曾在 21,000 秒 shell timeout 结束；不要无条件重跑整个长矩阵，拆成较短 ABBA。
6. Deep 在真实未追踪 B2 重放上分离局部计算与 collective。只有正确性通过再测多 batch 吞吐。
7. 在三个模型各自可用的配置上扫 G/M、副本和节点数；分母计入整个服务部署的全部卡，保留长短输出混合场景。多副本、HTTP、连续批处理和最优节点数尚未建立。

## 交接更新规则

每轮将源代码 SHA、实际终态、完整检查范围、性能口径和剩余问题写入新的 evidence 文件，随后更新 `recheck-status.json`。不要覆盖旧失败记录，也不要把仍在运行的记录标成成功。此交接不表示三模型优化目标已全部完成。
