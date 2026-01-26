# Qwen3 MoE 重构说明（infinicore 版）

本文记录如何将 `Qwen3MoeSparseMoeBlock` 从 torch 迁移到 infinicore 框架、缺失的算子列表，以及针对缺失算子的临时 Python/NumPy 实现方式。

## 重构思路
- **接口保持一致**：在 `python/infinilm/models/qwen3moe/qwen3moe.py` 中实现 `Qwen3MoeExperts`、`Qwen3MoeTopKRouter`、`Qwen3MoeSparseMoeBlock`，类名和调用方式与 torch 版本一致，便于替换。
- **参数类型迁移**：专家权重、路由权重使用 `infinicore.nn.Parameter` 存储，并通过 `infinicore.empty/zeros` 创建，保持设备与 dtype 可配置。
- **算子优先用 infinicore**：线性层调用 `infinicore.nn.functional.linear`，其余缺失的路由相关算子用 Python/NumPy 暂存。
- **返回值保持形状**：MoE block 输出 `(batch, seq, hidden_dim)` 的混合结果，以及 `(batch, seq, top_k)` 的路由得分，方便对齐原有行为。

## 缺失算子与临时实现
当前 infinicore 不具备以下 torch 常用算子，均在 `qwen3moe.py` 内用纯 Python/NumPy 模拟：

| 功能 | torch 对应 | 现状 | 临时方案 |
| --- | --- | --- | --- |
| Softmax | `torch.softmax` | 缺失 | `_softmax_np`：转 NumPy，按最后一维计算 softmax |
| Top-K | `torch.topk` | 缺失 | `_topk_np`：`argpartition` 找前 k，再排序 |
| One-Hot | `torch.nn.functional.one_hot` | 缺失 | `_one_hot_np`：`np.eye` 生成 |
| Scatter/Add | `index_add_` | 缺失 | `np.add.at` 在 token 维度累加 |
| Mask/筛选 | `where/nonzero` | 部分缺失 | 使用 `np.nonzero`/`np.where` 组合 |

辅助函数 `_tensor_to_numpy`、`_from_numpy_like` 负责在 infinicore Tensor 与 NumPy 之间桥接，保持 dtype/device 一致；若底层增加直接转换接口，可移除这些桥接。

## 关键模块说明
- **Qwen3MoeTopKRouter**：对输入做一次线性投影（infinicore），随后用 NumPy softmax + top-k 得到路由得分与专家索引，可选归一化。
- **Qwen3MoeExperts**：对命中专家的 token 做 gate/up 投影、激活（支持 `silu/swish`、`gelu`、`relu`），再 down 投影，并用 `np.add.at` 进行按 token 维度的累加。
- **Qwen3MoeSparseMoeBlock**：展平 batch/seq 维喂入 router，拿到 `routing_weights` 和 `selected_experts` 后调用 experts 聚合，最后 reshape 回原始形状并返回路由得分。

## 已知限制与后续优化
- 路由路径使用 NumPy，暂时会有 CPU 往返与性能损失；待 infinicore 提供 softmax/top-k/one-hot/scatter 等算子后可彻底移除这些 Python 分支。
- 激活函数目前覆盖 `silu/swish`、`gelu`、`relu`，若配置中包含其他激活需在 `_activation_fn` 扩展。
- 权重初始化沿用 `empty/zeros`，如需与原模型严格对齐，可在加载或构建阶段补充初始化逻辑。
- 建议后续补充单测，对比 torch 参考实现的输出形状与数值（在可用时）以确保兼容性。
