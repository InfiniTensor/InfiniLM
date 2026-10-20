# 实际通信路径与待优化点

本轮通过运行日志、实际加载库及节点计数器检查通信，而非仅检查启动参数。

| 当前作业 | 实际路径 | 结论 |
| --- | --- | --- |
| Kimi 原生 BF16/FP32 与 F64 参考 | `IBext_v8`，四个 `shca` 端口；已加载 SHCA 插件和配套 UCX | 使用高速互联；RCCL 明确报告 GPU Direct RDMA Disabled，仍有优化空间 |
| GLM 长输出 M8 | `NET/IB : No device found`，随后 `Socket`，接口为 `ib0–ib3` | TCP/IPoIB，未发现模型数据回退管理接口的证据；没有使用 RDMA 数据路径 |
| Deep TP8 | 单节点八卡 | 不能用它判断跨节点网络 |

检查到的每个高速端口均为 ACTIVE，报告 400 Gb/s NDR；这不是实测有效带宽。节点计数器采样为两秒且非进程归属统计，Kimi 部分任务处于权重加载阶段，管理接口存在流量不能单独证明 collective 走管理网络。

已另起两组 GLM 完整模型对照：一组显式指定 Socket 且接口限定 `ib`，一组强制 SHCA/IBext_v8，并使用同一 Model3072、G32/G64、M8、最多128输出协议。新任务须检查实际 backend 日志及完整 token、有限值、EOS、owner 和退出，再报告吞吐。第一轮使用不同节点组，之后应交叉节点，不能直接宣称因果加速比例。

数据通道、CPU 控制通道、权重存储流量和 GDR 能力应分别核实。仅有 TCP bootstrap 或 `NCCL_NET` 配置不构成数据面回退或高速生效的证据。详细机器映射留在本地私有快照；公开摘要见 `evidence/network-verification.json`。
