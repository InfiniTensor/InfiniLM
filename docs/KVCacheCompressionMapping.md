# KV Cache Compression Weight Mapping (llava_mlp.bin)

## 前缀与含义
权重来自 Fastcache 的 KVCacheLinearDecoupleCompressor，`.pth` 结构位于 `compressor` 子树，键模式为 `<prefix>.<layer>.<slot>.weight`。当前导出的 bin 包含以下前缀（已按排序写入）：

- `compress_tk`: 文本 K 压缩/投影相关权重
- `compress_tv`: 文本 V 压缩/投影相关权重
- `compress_iv`: 图像 V 压缩/投影相关权重（命名可能沿用 image/value 缩写）
- `compress_ik`: 图像 K 压缩/投影相关权重
- `attention`: 压缩器内部的注意力/门控线性层（小头数，通常 slot=0..7 等）

> 注：原始 PyTorch 权重中未见 bias；转换脚本若发现 bias 长度不匹配会跳过。

## 排序与写入顺序
排序键：`prefix` 优先级（`compress_tk` → `compress_tv` → `compress_iv` → `compress_ik` → `attention`），然后 `layer` 升序，再 `slot` 升序。同一 `(prefix,layer,slot)` 下先 weight 后 bias（若存在）。

## 形状推断与 hidden_size
- 头部的 `hidden_size` 来自首个权重的列数（当前为 640）。
- 每个 weight 块记录 `rows`、`cols`。可视为线性层 `out = W * in`，`W` 形状为 `[rows, cols]`，输出维度 = `rows`。
- bias（若存在且长度==rows）紧随其后。

## 可能的计算图猜测（供 C++ 实现对齐）
- `compress_tk`: 对文本 K 做降维/解耦，slot 多个表示分阶段或多头混合投影。
- `compress_tv`: 对文本 V 做降维/解耦。
- `compress_iv`: 对图像 V 做降维/解耦。
- `compress_ik`: 对图像 K 做降维/解耦。
- `attention`: 压缩器内部的小型注意力/门控 MLP，用于生成压缩映射或融合文本/图像特征。

实际计算顺序需结合 Fastcache 的 Python 源码（`KVCacheLinearDecoupleCompressor.forward`）逐层映射，将上述权重映射到具体的线性/激活/重排操作。

## 与 bin 对齐的校验
- 使用 `scripts/verify_llava_mlp_bin.py` 可对比 `.pth` 与 `.bin`：会打印头部、逐块形状及 max diff。
- 当前验证结果：`num_layers=32`，`weight_count_per_layer=12`，384 个 weight 块，max diff=0（fp16）。

