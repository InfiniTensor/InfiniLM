# KV Cache 压缩接入说明（InfiniLM 0.2.0 / csrc）

本文档说明 KV-cache 压缩在 InfiniLM 0.2.0（csrc 推理路径）中的接入位置，并给出与 PyTorch 参考实现的 1:1 对照，便于复核逻辑一致性。

> 重要限制
> - 仅支持 **普通 StaticKVCache**（连续布局）。
> - 不支持 **PagedKVCache / paged attention**。
> - 压缩假设 **image KV 是前缀**（若 image token 不在序列前缀，需自行避免或显式设置 image_kv_len=0）。

---

## 1. InfiniLM 接入点（csrc）

### 1.1 核心实现

- **KV 压缩实现**：`csrc/cache/kv_compression.cpp`
  - 读取 `.bin` 权重。
  - 按 `factor` 分组压缩 K/V（两条 3 层 MLP）。
  - 可选 image/text 前缀分段压缩。
- **StaticKVCache 扩展**：`csrc/cache/kv_cache.hpp` / `csrc/cache/kv_cache.cpp`
  - 新增 `StaticKVCache::compress_inplace()`。
- **模型侧入口**：
  - `csrc/models/infinilm_model.hpp` 新增虚接口 `compress_kv_cache_inplace`。
  - `csrc/models/llama/llama_for_causal_lm.cpp` 实现压缩调用。
  - `csrc/models/llava/llava_model.cpp` / `csrc/models/minicpmv/minicpmv_model.cpp` 转发到 Llama。
- **推理引擎层**：
  - `csrc/engine/rank_worker.cpp` 新增压缩任务。
  - `csrc/engine/infer_engine.cpp` 新增 `compress_kv_cache_inplace()`。
  - `csrc/pybind11/engine/engine.hpp` 导出 Python 接口。

### 1.2 Python 接入

- **配置对象**：`python/infinilm/cache/cache.py` 中新增 `KVCompressionConfig`。
- **推理调用**：`python/infinilm/infer_engine.py` 的 `generate()` 支持 `kv_compression_config`。
- **示例脚本**：`examples/jiuge.py` 增加 `--kv-compress` 参数组。

---

## 2. InfiniCore 新增算子（支撑多模态 csrc 模型）

为支持 LLaVA/MiniCPM-V 在 csrc 侧的视觉分支，新增/补齐以下算子：

- `layer_norm`（LayerNorm 模块）
- `quickgelu`（CLIP 视觉分支）
- `gelutanh`（SigLIP 视觉分支）
- `gelu / relu / softmax / conv2d`（视觉与投影所需）

这些算子保证视觉塔与多模态投影在 InfiniCore 中可编译、可运行。KV 压缩器本身仍使用 ReLU（与 PyTorch 参考一致）。

---

## 3. PyTorch 参考实现对照

参考实现：`Fastcache/utils_ccm/module_ccm_v11.py` 中的 `KVCacheLinearDecoupleCompressor`。

### 3.1 权重命名对照

| PyTorch 模块名 | 语义 | InfiniLM prefix 索引 | InfiniLM slot |
| --- | --- | --- | --- |
| compress_tk | 文本 Key MLP | 0 | 0/1/2 |
| compress_tv | 文本 Value MLP | 1 | 0/1/2 |
| compress_ik | 图像 Key MLP | 2 | 0/1/2 |
| compress_iv | 图像 Value MLP | 3 | 0/1/2 |

> `.bin` 中权重按 prefix->slot 顺序排列；每个 prefix 固定 3 层 Linear。

### 3.2 数据布局对照

| 项 | PyTorch | InfiniLM(csrc) |
| --- | --- | --- |
| KV 形状 | `[B, H, S, D]` | `[H, S, D]`（单 batch 视角） |
| 分段 | `it_len` 为图像前缀长度 | `image_kv_len` |
| 分组压缩 | `S' = floor(S / factor)` | 同样按 `factor` 分组 |
| 最小长度阈值 | `min_seq_len` | `min_seq_len` |

### 3.3 MLP 结构对照

PyTorch（训练版）：
```
Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear
```

InfiniLM（推理版）：
```
Linear -> ReLU -> Linear -> ReLU -> Linear
```

说明：推理端去掉 Dropout，三层 Linear 的权重与 PyTorch 对齐。

### 3.4 压缩流程对照（核心逻辑）

PyTorch（简化伪码）：
```
compressed_len = seq_len // factor
compress_len = compressed_len * factor
k_to_compress = k[:, :, :compress_len].reshape(B, H, compressed_len, D * factor)
compressed_k = compressor_k(layer)(k_to_compress)
remainder = k[:, :, compress_len:]
compressed_k = cat([compressed_k, remainder], dim=2)
```

InfiniLM csrc（等价步骤）：
```
compressed_len = seq_len // factor
compress_len = compressed_len * factor
k_head = k[:, :compress_len, :]                        # [H, S, D]
k_grouped = view(k_head, [H, S', D * factor])
k_in2d = view(k_grouped, [H * S', D * factor])
k_comp2d = MLP(k_in2d)
k_comp = view(k_comp2d, [H, S', D])
remainder = k[:, compress_len:, :]
concat([k_comp, remainder], dim=1)
```

---

## 4. 适配注意事项

- **PagedKVCache**：当前不支持；若后续适配，需要在 paged gather/scatter 上增加压缩前后长度映射。
- **image 前缀假设**：`image_kv_len` 只支持前缀长度；若 image token 不在前缀，建议设为 0。
- **多卡**：当前压缩实现默认单卡单批次逻辑，跨卡仍需额外调度。
