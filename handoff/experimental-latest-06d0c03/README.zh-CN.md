# 最新重构实验源码附加包

这是三模型最新实际构建源码的附加包，**不是第三套已经验收的稳定性能版本**。legacy/current 原有源码保留原始字节。

固定依赖：InfiniLM `06d0c03cc860d730afe5f35e41e53a443221da23`；InfiniCore `046f012a3ac0d2a91aede1791c0cebc47ed27b45`；InfiniOps `f998e4d6b445e29a950d048d0bc31f494ca7cce9`；InfiniRT `b4138168dfbbaa5e222febe66ab85919378ebc19`；InfiniCCL `6f329f83e6e248d9849b3ae1134dbaf05b409bc3`。

`cluster-layout/` 保留集群工作区相对路径，包含9套 Model/Common/Ops 实际源码、三模型 helpers 与构建配方、GLM PP2/大batch候选、Deep FP32候选、Kimi投机验证候选及 InfiniLoader 非主分支源。源码归档8,183个文件已逐一校验；`source-provenance.json` 记录归档SHA和公开占位符替换。无已编译对象、共享库、模型权重或私钥。

恢复时先按固定提交获取官方仓库及其递归依赖，再把所需隔离目录的源文件覆盖到对应新工作区，保留 Model/Common 的区别。第三方依赖从原交接的固定依赖说明或官方固定版本恢复，本附加包省略了vendor目录。不要把三个模型的私有 Runtime/Model 布局交叉链接。

在新环境先重建同提交RT/CCL，随后构建各模型Ops SDK、Common Python/Runtime、GPU helpers和Model；参考 `drivers/*r3550*`、`common_latest_r3551_inside.sh`、`build_latest_models_r3586_inside.sh`。历史脚本中的 `/refactor`、`/old`、`/shared` 是容器内绑定约定；旧机器的源/构建 receipt 和二进制SHA不能直接当作新构建结果。设置真实路径、编译器、CPU affinity及网络，生成本机新 receipt，重新完成算子、B1 QA和对应batch完整token验收。不要复制旧 `.so` 或仅删除校验断言绕过依赖检查。

投机入口是默认关闭的 `INFINILM_KIMI_SPEC_R3632=1` 和专用CAPI，当前仅实验驱动支持；不是通用HTTP服务开关。先读 [投机实验说明](../SPECULATIVE_EXPERIMENT.zh-CN.md)。主线加载、CPU缓存、拓扑能力和大B/M状态见 [并行与加载](../LOADING_AND_PARALLELISM.zh-CN.md)。新服务器须重新发现资源及通信，不能沿用原作业号、节点占位符或显存假设。
