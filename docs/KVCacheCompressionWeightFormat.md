# KV Cache 压缩权重二进制格式（`.bin`）

本文档描述 `scripts/convert_kv_compressor_pth_to_bin.py` 生成、并由 `Compressor::loadWeights()` 读取的 KV-cache 压缩权重二进制格式（当前版本：`v1`）。

> 约定：所有整数均为 **little-endian**。

---

## 1. 文件结构

```
| Header (44 bytes) |
| metadata (optional, metadata_size_bytes bytes) |
| For layer in [0, num_layers):
|   For i in [0, weight_count_per_layer):
|     WeightMeta (12 bytes)
|     weight_data (rows * cols * sizeof(dtype))
|     [bias_data] (rows * sizeof(dtype))  # only if has_bias==1
```

---

## 2. Header（固定 44 字节）

`Header` 对应 Python `struct.pack("<IIHHIIIIIIII", ...)` 与 C++ 结构体：

- `magic` (`uint32`): 固定为 `0x4B56434D`
- `version` (`uint32`): 当前为 `1`
- `dtype_code` (`uint16`): 权重数据类型编码
  - `0`: fp16（IEEE float16，小端）
  - `1`: bf16（以 `uint16` 原始 payload 存储，与 `torch.bfloat16` storage 一致）
  - `2`: fp32（IEEE float32，小端）
- `reserved` (`uint16`): 保留（写入 0）
- `num_layers` (`uint32`): 层数（与模型 transformer layers 对齐）
- `num_heads` (`uint32`): 可选字段（目前不强依赖，允许为 0）
- `head_dim` (`uint32`): 头维度（可选 sanity 字段）
- `hidden_size` (`uint32`): 隐藏维度（可选 sanity 字段）
- `compression_factor` (`uint32`): 压缩因子（例如 `5`）
- `min_seq_len` (`uint32`): 触发压缩的最小长度阈值
- `weight_count_per_layer` (`uint32`): 每层权重块数量（不包含 bias 的额外块；bias 由 `WeightMeta.has_bias` 决定是否跟随）
- `metadata_size_bytes` (`uint32`): metadata 区大小（当前转换脚本写入 0）

---

## 3. WeightMeta（每个权重块 12 字节）

每个权重块以 `WeightMeta` 开头：

- `rows` (`uint32`)
- `cols` (`uint32`)
- `has_bias` (`uint32`): `1` 表示紧随其后有 bias 数据；`0` 表示无 bias

紧随其后的数据：

- `weight_data`: `rows * cols` 个元素
- `bias_data`（可选）: `rows` 个元素（当 `has_bias==1`）

---

## 4. 权重顺序（per-layer order）

当前实现按「每层」顺序存放压缩器 MLP 的线性层权重。默认导出顺序由转换脚本参数决定：

- `--prefix-order` 默认：`compress_tk,compress_tv,compress_ik,compress_iv`
- `--slot-order` 默认：`0,3,6`（对应 `.pth` 里每个 prefix 的 3 个 `Linear` 层）

因此默认情况下：

- 每层有 `4(prefix) * 3(slot) = 12` 个 `WeightMeta + weight_data(+bias)` 块
- 若导出 text-only（只含 `compress_tk/compress_tv`），则每层为 `2 * 3 = 6` 个块

注意：运行时会按「prefix->slot」顺序取用，因此导出顺序必须与加载端一致。

---

## 5. 生成方式

推荐直接使用脚本生成（支持输出 `fp16/bf16/fp32`）：

```bash
python3 scripts/convert_kv_compressor_pth_to_bin.py \
  --input path/to/compressor.pth \
  --output ./compress_ckpt/compressor.bin \
  --dtype fp16
```

该脚本会把 `.pth` 中的压缩器 state_dict 展开为上述二进制格式（无需 PyTorch 运行时依赖即可被 InfiniLM 读取）。

