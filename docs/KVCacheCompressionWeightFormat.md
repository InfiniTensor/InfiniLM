# KV Cache Compression Weight Format (Binary, No PyTorch Dependency)

## File Layout
All values are little-endian unless stated otherwise. Strings are ASCII, null-terminated.

- Header (fixed size)
  - `uint32` magic = 0x4B56434D ("KV C M")
  - `uint32` version = 1
  - `uint16` dtype code: 0 = fp16, 1 = bf16, 2 = fp32
  - `uint16` reserved = 0
  - `uint32` num_layers
  - `uint32` num_heads
  - `uint32` head_dim
  - `uint32` hidden_size
  - `uint32` compression_factor (e.g., 4, 5)
  - `uint32` min_seq_len
  - `uint32` weight_count_per_layer (for sanity check)
  - `uint32` metadata_size_bytes (future expansion; set 0 for now)
- Layer blocks (repeat `num_layers` times)
  - For each weight tensor (order defined below):
    - `uint32` rows
    - `uint32` cols
    - `uint32` has_bias (0/1)
    - data blob for weight: `rows * cols * sizeof(dtype)`
    - optional bias blob: `cols * sizeof(dtype)` when `has_bias==1`
- Footer
  - `uint32` checksum (optional; set 0 if not used)

## Weight Order per Layer (example for linear-decouple MLP)
Adjust if实际模型结构不同，但顺序需在导出和加载一致。
1. `proj_k` weight (+bias)
2. `proj_v` weight (+bias)
3. `compress_k` weight (+bias)
4. `compress_v` weight (+bias)
5. `decompress_k` weight (+bias)
6. `decompress_v` weight (+bias)
7. `gate`/`mlp` weights (+bias) if算法需要

`weight_count_per_layer` = 实际包含的权重项数，便于解析时校验。

## Export Steps (one-time, in external Python env)
1) 使用 PyTorch 读取原 `.pth`：`state = torch.load(...)`.
2) 提取压缩器权重到固定顺序的列表；统一 dtype（fp16/bf16）：
   ```python
   weights = [
     (state['proj_k.weight'], state.get('proj_k.bias')),
     ...
   ]
   ```
3) 写入头部；逐层写元信息 + 数据；按 `dtype` 转为字节（fp16 用 `np.float16.tobytes()`）。
4) 填充 footer（可置 0）。

## Loader Expectations (C++/InfiniCore)
- 读取并验证 magic/version/dtype/层数/weight_count_per_layer。
- 为每个权重创建 `Tensor::weight`，dtype 与头部一致。
- 如果缺少某些权重（has_bias=0），按约定跳过 bias。
- 解析出的权重按同样顺序存入压缩器对象，以确保前向逻辑正确。
