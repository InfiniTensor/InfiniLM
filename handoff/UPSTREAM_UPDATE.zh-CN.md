# 最新上游迁移状态

本轮观测到用户指定分支更新为 [c3deba4](https://github.com/InfiniTensor/InfiniLM/commit/c3deba450a631d79e3a183048726b4a1f4acabdd)，Core 为 `14a8bd52b586c26fcfdf7ecc04efde38640467fb`，Ops 为 `bcfe96daf338b9f9992049f7965b27b6afe88b68`；RT `b4138168dfbbaa5e222febe66ab85919378ebc19`、CCL `6f329f83e6e248d9849b3ae1134dbaf05b409bc3` 不变。

`upstream-update-c3deba4/recursive-source-manifest.json` 保存精确递归 gitlink 和源文件验证信息。该目录保留下载、源合并、Pad 冲突解决和 SDK 构建脚本，作为继续迁移的工程输入；它不是额外一套已验收的运行时。下载器的历史缓存路径在新服务器可不存在，应按精确提交重新获取，不能替换为浮动 HEAD。

上游统一 Mul/MulScalar，Pad 取代 PadConstant。私有 Hygon Cast64、归约、生命周期和参数校验需要保留；9 套 Ops/Model/Common 源已合并，5 处 Pad 冲突显式解决。三个 SDK 编译安装后在 CPU-only 加载阶段 abort，尚无 GPU 算子、Common、Model 或性能验收。后续先保留失败首错，在隔离进程核对 Torch 预加载和依赖，再完整重建新 API 使用者。

当前交接的稳定性能仍锁定 `3e0d89fe`。不要将旧版 117/63/26 或当前 GLM/DeepB4 的结果标到本次新提交下。

更新至本轮末：新上游三个 SDK 已在健康节点通过 GPU 可见加载（每节点8卡小计算检查）；原 CPU-only 加载失败是无可见 ROCm 设备，保留负例。新 Common/Model、算子与吞吐仍须验收。旧版 Kimi 长文已独立审计 68.53–68.70/59.89–60.10 tok/s；当前 GLM G128/M32、最多128输出的组合优化已审计 60.51/20.83 tok/s（decode/含prefill），详见性能末表。Deep B8 分块 prefill 已重新测试；Kimi 流水线 producer 已结束，独立审计待完成。
