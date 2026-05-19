# Hybrid Attention on Iluvatar

本文档记录 Iluvatar 平台上的 `hybrid-attn` 路径：Prefill 使用 FlashAttention-2 varlen，Decode 使用 InfiniCore 原生 PagedAttention。

## 改了什么

### InfiniCore

- Iluvatar 被作为 CUDA-compatible ATen device 处理，用于复用 ATen/CUDA tensor 和 stream guard。
- Iluvatar 的 FlashAttention-2 调用走全局 `flash_attn_2_cuda` ABI。
- 适配 Iluvatar 当前 `mha_varlen_fwd` 尾部参数。
- `mha_varlen` 和 `mha_kvcache` 的 FlashAttention 路径使用当前 InfiniCore stream。
- Iluvatar + `--flash-attn` 构建时，`libinfinicore_cpp_api.so` 链接 `flash_attn_2_cuda*.so` 并写入 rpath。
- 构建时同步 PyTorch 的 `_GLIBCXX_USE_CXX11_ABI`，避免 Python 扩展加载时 ABI 符号不匹配。

### InfiniLM

- 新增 `hybrid-attn` attention backend。
- 新增独立 `HybridAttentionImpl`，不改变纯 `flash-attn` 语义。
- `hybrid-attn` 的执行路径：
  - Prefill：使用 FA2 varlen，输入为本轮 dense `query/key/value`。
  - Decode：使用原生 `paged_attention_`，输入为 paged KV cache。
- `hybrid-attn` 使用 paged KV cache 分配。
- Python CLI/API 层会把 `hybrid-attn` 归一化为 paged cache 路径，避免 cache 类型误配。

## 怎么使用

以下命令以 `/data-aisoft/qyq_models/Qwen2.5-3B-Instruct` 为例。

### 1. 环境变量

FA2 的 `.so` 路径可以直接通过 Python 获取：

```bash
export FLASH_ATTN_2_CUDA_SO=$(python3 -c 'import flash_attn_2_cuda; print(flash_attn_2_cuda.__file__)')
export LD_LIBRARY_PATH=/root/.infini/lib:/usr/local/corex/lib64:/usr/local/corex/lib64/python3/dist-packages/torch/lib:/usr/local/corex/lib64/python3/dist-packages:$LD_LIBRARY_PATH
export PYTHONPATH=/home/zx/InfiniLM/python:/home/zx/InfiniCore/python:/usr/local/corex/lib64/python3/dist-packages:$PYTHONPATH
```

### 2. 构建 InfiniCore

```bash
cd /home/zx/InfiniCore
xmake f --iluvatar-gpu=y --aten=y --flash-attn=/usr/local/corex/lib64/python3/dist-packages
xmake build infinicore_cpp_api
xmake build _infinicore
xmake install -o /root/.infini infinicore_cpp_api
xmake install -o /root/.infini _infinicore
```

同步本地 Python 包中的 InfiniCore 扩展：

```bash
cp -f /root/.infini/lib/libinfinicore_cpp_api.so /home/zx/InfiniCore/python/infinicore/lib/libinfinicore_cpp_api.so
cp -f /root/.infini/lib/_infinicore.cpython-310-x86_64-linux-gnu.so /home/zx/InfiniCore/python/infinicore/lib/_infinicore.cpython-310-x86_64-linux-gnu.so
```

可选检查 `libinfinicore_cpp_api.so` 是否已经链接 FA2：

```bash
readelf -d /root/.infini/lib/libinfinicore_cpp_api.so | grep flash_attn_2_cuda
```

预期能看到类似：

```text
NEEDED Shared library: [/usr/local/corex/lib64/python3/dist-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so]
```

### 3. 构建 InfiniLM

```bash
cd /home/zx/InfiniLM
xmake build _infinilm
xmake install _infinilm
```

### 4. 运行 hybrid-attn 推理

```bash
cd /home/zx/InfiniLM
python3 examples/test_infer.py \
  --model /data-aisoft/qyq_models/Qwen2.5-3B-Instruct \
  --device iluvatar \
  --enable-paged-attn \
  --attn hybrid-attn \
  --batch-size 1 \
  --max-new-tokens 4 \
  --prompt "你好" \
  --temperature 0.0 \
  --top-k 1
```

说明：当前 CLI/API 会将 `hybrid-attn` 归一化到 paged cache 路径；命令中保留 `--enable-paged-attn` 是为了显式表达运行条件。

## Qwen2.5 运行结果

验证环境：

- Platform：Iluvatar
- Model：`/data-aisoft/qyq_models/Qwen2.5-3B-Instruct`
- Attention backend：`hybrid-attn`
- Batch size：1
- Max new tokens：4
- Prompt：`你好`

构建验证：

```text
xmake build infinicore_cpp_api    # passed
xmake build _infinicore           # passed
xmake build _infinilm             # passed
```

推理复现结果：

```text
load weights over! 2431.8737983703613 ms

=================== start generate ====================
Generating: 100%|██████████| 1/1 [00:02<00:00,  2.53s/it]
Resquest 0:
===Query===
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant

===Response===
""""

total_time: 2582.32 ms
```

## 当前边界

- 当前稳定验证路径是 Iluvatar + Qwen2.5 + FA2 dense prefill + native paged decode。
- Iluvatar 当前 FA2 varlen 不使用 paged KV cache layout 作为 prefill 输入，hybrid prefill 使用本轮 dense `key/value`。
- `flash-attn` 仍表示纯 FA 路径；`hybrid-attn` 是单独 backend。
