# KV Cache Compression Integration Plan (No PyTorch)

## 目标与约束
- 在 InfiniLM 内集成 KV 压缩，减少显存占用，保持吞吐；仅使用现有 C++/InfiniCore 栈，不引入 PyTorch 依赖。
- 支持可选开关，默认关闭；压缩失败或序列过短时可回退原生 KV。
- 兼容多设备分片（ndev>1），不破坏现有 API/线程模型。

## 现状扫描
- KV 结构：`KVCache` 以 `Tensor` 存 K/V，按设备与层分片；创建在 `cache_manager/kvcache.cpp`。
- 执行：每设备线程在 `inferDeviceBatch` 里直接读写 KV，使用 `MemoryPool` 分配中间 buffer。
- Python 侧：ctypes 入口只传裸指针，当前无压缩相关配置。
- 现有 `Fastcache` 提供压缩思路（线性解耦/MLP），但是 PyTorch 实现，不可直接依赖。

## 设计方向
1) **压缩格式抽象**
   - 定义 `CompressedKV`：包含每设备每层的压缩后张量、长度元信息（原始 seq_len、压缩后 seq_len、索引映射/权重）。
   - 统一接口：`compress(KVCache&, meta) -> CompressedKV`，`decompress(CompressedKV&, tmp_buffers, positions)` 或在注意力前做“按需解压”。
2) **调用流程插桩**
   - Prefill 完成后（或到达阈值 seq_len）调用 `compress`；释放/替换原 KV（或标记仅持有压缩版）。
   - 解码阶段，在注意力前决定：若启用压缩则从 `CompressedKV` 解压到临时 buffer；否则走原路径。
   - 线程模型保持不变：每设备线程独立压缩/解压自身分片；rank0 仍负责最终 logits/采样。
3) **算法简化与实现策略（无 PyTorch）**
   - **阶段性方案 A（快路径）**：基于采样/截断的轻量压缩（如“保留最近 N + 按重要性采样历史”），无需 MLP 权重；便于快速验证链路与收益。
   - **方案 B（完整解耦 MLP 移植）**：将 Fastcache 的线性解耦/MLP 转写为 C++/InfiniCore 运算：
     - 解析 `.pth`：用 minimal torch-less 读取（如 LibTorch? 不允许则要求提供原始权重二进制），或定义自有权重格式。
     - 计算图：实现线性层、激活（SiLU/GELU）、分块重排；支持 FP16/BF16。
     - 需要新增算子支持时，优先复用已有 InfiniCore 线性/elemwise；否则补充自定义 kernel。
   - **长度映射**：保留/生成 indices 或投影矩阵，供解压或稀疏注意力使用；考虑存储为 int32 索引 + 归一化权重。
4) **数据结构与内存**
   - `CompressedKV` 使用 `MemoryPool` 分配压缩张量与索引；生命周期与 `KVCache` 同步。
   - 临时解压 buffer 同样从 `MemoryPool` 申请，避免频繁 malloc/free。
   - 支持 per-layer 不同压缩率（可统一默认因子）。
5) **API 与配置**
   - C API 扩展：在 `createJiugeModel` 增加压缩配置（启用、压缩因子、最小 seq_len、算法类型、ckpt 路径占位）。
   - Python 绑定更新相应字段，脚本添加 CLI 参数；默认关闭。
   - 保持现有 KV API 不变，压缩接口内部管理。
6) **多设备与同步**
   - 每设备独立压缩/解压各自 head 分片；不需要跨设备通信变更。
   - NCCL all-reduce 仍在 logits/FFN 后执行；压缩不影响现有同步点。
7) **错误与回退**
   - 压缩条件不满足（序列过短、内存不足、算子不支持）时记录日志，回退未压缩路径。
   - 提供开关跳过解压路径，用于 A/B 测试。

## 迭代计划
- **阶段 1：接口与占位实现**
  - 定义 `CompressedKV` 结构、压缩配置、C API/绑定扩展；实现方案 A（简单截断/采样压缩），验证链路与内存下降。
- **阶段 2：算子与解耦逻辑**
  - 移植 Fastcache 线性解耦核心到 C++/InfiniCore，支持 FP16/BF16；实现压缩/解压闭环。
- **阶段 3：集成与测试**
  - 在 `inferDeviceBatch` 插入压缩/解压路径，完善回退。
  - 编写最小示例脚本（纯 ctypes）对比开启/关闭压缩的输出一致性、显存占用、吞吐。
- **阶段 4：优化与文档**
  - 调优内存池尺寸、压缩率策略（动态阈值）。
  - 文档化配置与行为，增加监控/日志。

## 风险与待决
- `.pth` 权重解析若不能依赖 PyTorch，需要定义替代格式或预导出二进制。
- InfiniCore 是否已有所需激活/线性/重排算子，缺失部分需新增 kernel。
- 压缩后精度影响待评估：需提供一致性检查与可配置回退。

## 详细实施步骤（可按此执行落地）

### 0) 准备阶段
- 确认压缩器算法选择：以 Fastcache 的 KVCacheLinearDecoupleCompressor 为目标（解耦 MLP），但权重格式改为自定义二进制。
- 确认运行约束：不依赖 PyTorch/LibTorch；仅使用 C++17 + InfiniCore。

### 1) 权重导出与加载格式
- 定义权重文件格式（建议单文件二进制）：
  - 头部：魔数 + 版本号 + dtype（fp16/bf16/fp32）+ 压缩因子 + 层数 + hidden 维度 + head 数。
  - 按序存储每层权重：线性层权重、bias、必要的缩放因子；采用行主序，dtype 与头部一致。
  - 可选：索引表/元信息（若固定结构，可省）。
- 在 Python 环境中编写一次性导出脚本（使用 torch 读取 .pth → 按上述格式写 bin）；保存在 Fastcache/ckpt 之外。
- 在 C++ 侧编写解析器：读取文件头，校验版本/dtype/尺寸，然后逐层用 `Tensor::weight` 创建权重张量。

### 2) 数据结构与配置
- 新增结构：
  - `CompressionConfig`：启用开关、压缩因子、最小 seq_len、权重路径、适用层范围。
  - `CompressedKV`：每设备/每层压缩后的 K/V 张量、压缩后 seq_len、原 seq_len、索引/权重等元数据。
  - `Compressor` 类：持有权重张量和配置，提供 `compress`/`decompress`。
- 将 `CompressionConfig` 挂到模型（如 `JiugeModel`）实例；默认关闭。

### 3) 算子与计算图实现
- 复用现有 InfiniCore 运算：
  - 线性层（矩阵乘 + bias）：使用已有 `linear` 包装；确认支持输入/输出 dtype（fp16/bf16）。
  - 激活：SiLU/GELU；若缺失，补充简单元素级 kernel。
  - 重排/拼接：根据压缩因子将 seq 维分块或投影；可用 `Tensor` 的 view/slice/permute。
- 设计算法流程（对每层 K/V）：
  - 输入：K/V [B, nhead, seq, dim]（按设备分片）。
  - 可能的操作：先重排维度 → 线性映射降维或子采样 → 存储压缩后的 Kc/Vc 及映射。
  - 输出：压缩后的 Kc/Vc + 索引/scale，用于解压或稀疏访问。
- 解压路径：
  - 输入压缩 Kc/Vc，利用索引/映射还原到临时 buffer [B, nhead, seq', dim]（seq' = 原 seq 或压缩 seq）。
  - 供注意力使用，完成后由 MemoryPool 释放。

### 4) 生命周期与调用点改造
- 模型初始化（`JiugeModel` 构造）：
  - 若启用压缩：加载权重文件，实例化 `Compressor`（每设备共享或每设备一份视需要而定）。
- Prefill 后：
  - 判断 `seq_len >= min_seq_len` 且开启压缩 → 调用 `compress(KVCache&, cfg)` 生成 `CompressedKV`，可选择释放原 KV 或保留引用。
- 解码时（`inferDeviceBatch` 内部）：
  - 若存在 `CompressedKV`：解压到临时 K/V 张量（MemoryPool），用临时 K/V 做注意力。
  - 若未压缩或回退：直接使用原 KV。
- 线程模型保持不变：每设备线程持有自己的 `CompressedKV` 与临时 buffer。

### 5) C API 与绑定
- 扩展 `createJiugeModel` 参数：增加压缩开关、压缩因子、最小序列长度、权重路径字符串。
- ctypes 层（Python 绑定）同步增加结构字段，但默认关闭，避免破坏现有调用。
- 暂不提供 PyTorch 依赖的压缩调用，保持纯 C API。

### 6) 测试与验证（无 PyTorch）
- 编写最小 C++ 测试（可选通过 ctypes 调用）：
  - 构造假 KV（随机数据）→ 调用 `compress` → `decompress` → 与原始 KV 做点对点校验/误差容忍。
  - 测量内存占用：记录压缩前后 Tensor 大小，验证压缩因子。
  - 测量时间：压缩+解压开销；确保不阻塞主线程。
- 集成测试：在现有推理脚本中开启压缩开关，验证输出是否合理（允许轻微数值偏差）。

### 7) 性能与回退策略
- 若压缩后收益不足（seq 短）或压缩失败，记录日志并回退原 KV。
- MemoryPool 调整：根据压缩因子估计临时 buffer 大小，必要时扩容。
- 配置开关可快速关闭压缩以做 A/B 对比。

### 8) 里程碑拆解
- M1：权重格式确定 + 解析器完成；`CompressionConfig`/`CompressedKV` 定义完毕。
- M2：基础压缩/解压算子实现（截断/采样占位），插入调用点，链路可跑通并节省显存。
- M3：解耦 MLP 算法实现 + 权重加载验证，完成形状/数值校验。
- M4：集成测试（多设备、长序列），性能评估与回退机制完善。
- M5：文档与示例脚本更新，默认关闭压缩，配置项可控制。
