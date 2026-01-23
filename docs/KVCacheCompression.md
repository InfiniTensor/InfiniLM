# KV Cache 压缩（InfiniLM 0.2.0 / csrc）

本文档记录 KV-cache 压缩在 InfiniLM 0.2.0（csrc 推理路径）上的使用方式、编译步骤、单测与吞吐评测。

> 能力边界
> - ✅ 支持 StaticKVCache（连续布局）in-place 压缩
> - ✅ 已适配模型：`/data/huggingface/llava-1.5-7b-hf`、`/data/huggingface/MiniCPM-V-2_6`
> - ✅ NVIDIA 后端
> - ❌ 不支持 PagedKVCache / paged attention

---

## 1. 构建（NVIDIA）

### 1.1 编译 InfiniCore

```bash
cd /home/zhujianian/131/InfiniLM2.0/InfiniCore
xmake f -m release --nv-gpu=y
xmake -j$(nproc)
xmake install
```

### 1.2 编译 InfiniLM

```bash
cd /home/zhujianian/131/InfiniLM2.0/InfiniLM
xmake f -m release
xmake -j$(nproc)
xmake install
```

### 1.3 运行环境变量

```bash
export PYTHONPATH=/home/zhujianian/131/InfiniLM2.0/InfiniCore/python:/home/zhujianian/131/InfiniLM2.0/InfiniLM/python
export LD_LIBRARY_PATH=/home/zhujianian/miniconda3/envs/sdmllm/lib:/home/zhujianian/131/InfiniLM2.0/InfiniCore/python/infinicore/lib:/home/zhujianian/.infini/lib
```

> 如需将构建产物安装到 Python 包目录，可使用：
> `xmake install -o python/infinicore _infinicore` 与 `xmake install -o python/infinilm _infinilm`。

---

## 2. 压缩权重准备（.pth -> .bin）

KV 压缩只支持 `.bin` 权重。使用脚本转换 `.pth`：

```bash
cd /home/zhujianian/131/InfiniLM2.0/InfiniLM
python3 scripts/convert_kv_compressor_pth_to_bin.py \
  --input /home/zhujianian/cvpr/ckpt_store/best_finetune_mlp_1030_mm_9.pth \
  --output ./compress_ckpt/llava_mlp_local.bin \
  --dtype fp16

python3 scripts/convert_kv_compressor_pth_to_bin.py \
  --input /home/zhujianian/cvpr/ckpt_store/best_finetune_mlp_13B_mm_1_minicpm.pth \
  --output ./compress_ckpt/minicpmv_mlp_local.bin \
  --dtype fp16
```

本地可生成 `.bin` 示例：`compress_ckpt/llava_mlp_local.bin`、`compress_ckpt/minicpmv_mlp_local.bin`（不建议提交到仓库）。

---

## 3. 推理使用（示例：examples/jiuge.py）

压缩只在 **StaticKVCache** 下生效，请不要使用 `--enable-paged-attn`。

### 3.1 LLaVA

```bash
cd /home/zhujianian/131/InfiniLM2.0/InfiniLM
python3 examples/jiuge.py \
  --nvidia \
  --model_path /data/huggingface/llava-1.5-7b-hf \
  --image /home/zhujianian/cvpr/wuhang/bus.jpg \
  --prompt "Describe this image." \
  --max_new_tokens 64 \
  --kv-compress \
  --kv-compress-weight ./compress_ckpt/llava_mlp_local.bin \
  --kv-compress-factor 5 \
  --kv-compress-min-seq 2
```

说明：
- 若需要区分 image/text 前缀，可额外传 `--kv-image-kv-len`（LLaVA 1.5 默认单图为 576）。
- 若不确定 image 是否为前缀，建议保持 `--kv-image-kv-len 0`（仅用 text MLP）。

### 3.2 MiniCPM-V

```bash
cd /home/zhujianian/131/InfiniLM2.0/InfiniLM
python3 examples/jiuge.py \
  --nvidia \
  --model_path /data/huggingface/MiniCPM-V-2_6 \
  --image /home/zhujianian/cvpr/wuhang/bus.jpg \
  --prompt "图片是什么？" \
  --max_new_tokens 64 \
  --kv-compress \
  --kv-compress-weight ./compress_ckpt/minicpmv_mlp_local.bin \
  --kv-compress-factor 5 \
  --kv-compress-min-seq 2 \
  --kv-image-kv-len 0
```

说明：
- `examples/jiuge.py` 固定 `max_slice_nums=1`，保证单图单 slice，便于对齐 `image_bound`。
- MiniCPM-V 的 image token block 长度为 `query_num=64`，如需前缀压缩可设置 `--kv-image-kv-len 64`。

---

## 4. 单测（与原 PR 一致）

### 4.1 编译

```bash
cd /home/zhujianian/131/InfiniLM2.0/InfiniLM/tests
g++ -std=c++17 -O2 -I. -I.. -I../include -I../src -I${HOME}/.infini/include \
  -L${HOME}/.infini/lib -Wl,-rpath,${HOME}/.infini/lib -pthread \
  test_kv_compression_load.cpp -linfinicore_infer -linfiniop -linfinirt -linfiniccl \
  -o test_kv_compression_load

g++ -std=c++17 -O2 -I. -I.. -I../include -I../src -I${HOME}/.infini/include \
  -L${HOME}/.infini/lib -Wl,-rpath,${HOME}/.infini/lib -pthread \
  test_kv_compression_correctness.cpp -linfinicore_infer -linfiniop -linfinirt -linfiniccl \
  -o test_kv_compression_correctness

g++ -std=c++17 -O2 -I. -I.. -I../include -I../src -I${HOME}/.infini/include \
  -L${HOME}/.infini/lib -Wl,-rpath,${HOME}/.infini/lib -pthread \
  test_kv_compression_correctness_cpu.cpp -linfinicore_infer -linfiniop -linfinirt -linfiniccl \
  -o test_kv_compression_correctness_cpu
```

### 4.2 运行

```bash
cd /home/zhujianian/131/InfiniLM2.0/InfiniLM/tests
./test_kv_compression_load ../compress_ckpt/llava_mlp.bin nvidia
./test_kv_compression_correctness ../compress_ckpt/llava_mlp.bin nvidia
./test_kv_compression_correctness_cpu ../compress_ckpt/llava_mlp.bin
```

测试数据在 `tests/dump_kv/` 下：`meta.json`、`input_kv.bin`、`output_kv.bin`。

---

## 5. 吞吐评测（真实图片 & batch 对比）

评测脚本基于 `examples/jiuge.py`。
真实图片集参考：`/home/zhujianian/cvpr/llava_testcon2.6.py`。

统一设置：
- `max_new_tokens=32`
- `--no-stop-on-eos`（避免提前停止影响吞吐）
- 单图：`/home/zhujianian/cvpr/wuhang/bus.jpg`
- 压缩参数：`factor=5`、`min_seq=2`、`image_kv_len=0`

### 5.1 LLaVA（llava-1.5-7b-hf）

| bs | baseline prefill (tok/s) | baseline decode (tok/s) | compress prefill (tok/s) | compress decode (tok/s) | 备注 |
| --- | --- | --- | --- | --- | --- |
| 1 | 86.56 | 83.17 | 62.27 | 80.21 | - |
| 32 | 359.68 | 1574.21 | 309.65 | 2058.64 | - |
| 64 | 378.02 | 2294.92 | 319.57 | 3508.29 | - |
| 128 | OOM | OOM | OOM | OOM | cudaMalloc |

### 5.2 MiniCPM-V（MiniCPM-V-2_6）

| bs | baseline prefill (tok/s) | baseline decode (tok/s) | compress prefill (tok/s) | compress decode (tok/s) |
| --- | --- | --- | --- | --- |
| 1 | 427.43 | 87.53 | 364.16 | 88.48 |
| 8 | 1118.72 | 651.48 | 1030.15 | 664.39 |
| 16 | 1232.59 | 1233.87 | 1164.00 | 1254.53 |
| 32 | 1339.24 | 2335.95 | 1270.91 | 2368.02 |
| 64 | 1370.84 | 4207.61 | 1290.92 | 4223.16 |

---

## 6. 适配改动摘要（0.2.0）

- InfiniLM（csrc）：新增 `KVCompressionConfig`、`StaticKVCache::compress_inplace`、`InferEngine.compress_kv_cache_inplace`。
- Python：`InferEngine.generate` 支持 `kv_compression_config`。
- InfiniCore：新增 LayerNorm 与多种激活算子（QuickGELU/GELUTanh 等）以支持多模态视觉分支。
