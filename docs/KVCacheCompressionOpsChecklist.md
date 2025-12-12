# KV Cache Compression 算法拆解与算子需求（llava_mlp.bin 基线）

## 模块拆解（基于权重前缀推断）
- `compress_tk`: 文本 K 路径压缩/解耦。若 slot 多个，可能对应多阶段或多头混合投影。
- `compress_tv`: 文本 V 路径压缩/解耦。
- `compress_iv`: 图像 V 路径压缩/解耦。
- `compress_ik`: 图像 K 路径压缩/解耦。
- `attention`: 压缩器内部的小型注意力/门控线性层（可能用于融合/生成映射）。

## 可能的计算流程（参考 Fastcache 思路，需结合源码逐条对齐）
1) 对 KV 按类别（文本/图像）分支，按头/slot 做线性变换降维或投影。
2) 可选的 gating/注意力：使用 `attention.*` 权重对压缩特征做融合或生成索引/权重。
3) 生成压缩后的 K/V（seq 维缩短或维度降维），并记录映射（indices/scale）。
4) 解压路径：根据保存的映射/scale，将压缩 K/V 恢复到注意力可消费的形式（或直接在注意力中使用压缩格式）。

## 需要的核心算子（优先复用 InfiniCore）
- 矩阵乘 + bias：`linear`（已有）。需支持 fp16/bf16。
- 激活：SiLU/GELU（确认是否已有；缺失则补充逐元素 kernel）。
- 张量重排：view/reshape/permute/slice（`Tensor` 已支持）。
- 可能的归一化/缩放：元素级乘加（已有基础算子可组合）。
- 可选：索引/聚合（若压缩逻辑需要采样或根据权重重排 seq）。

## 需要明确的映射关系（待确认）
- 每个 prefix 对应的输入/输出形状：`[B, heads, seq, dim]` → `[...]`，slot 如何映射。
- 压缩因子作用位置：seq 维还是隐藏维（或两者结合）。
- `attention` 权重的输入特征来源与输出用途（生成哪类权重/索引）。
- 是否需要存储 indices/scale 以便解压或稀疏注意力。

## 实现阶段建议
1) **占位压缩**：先实现保留最近 N/截断的简版，打通链路。
2) **权重映射对齐**：阅读 Python 版 `KVCacheLinearDecoupleCompressor.forward`，写出每个 prefix 的线性/激活顺序和张量维度。
3) **算子补齐**：如缺 SiLU/GELU，新增简洁 kernel；其余用现有 linear/elemwise 组合。
4) **解压策略**：选择解压到密集 KV（改动小）或直接改注意力支持压缩格式（二选一）。
5) **验证**：构造 C++/ctypes 小测试，随机 KV → 压缩 → 解压 → 对比误差，量化开销与收益。
