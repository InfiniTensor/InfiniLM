# KV Cache 压缩（InfiniLM 0.2.0 适配记录）

本文档记录本仓库在 InfiniLM 0.2.0 上接入「KV-cache 压缩」的整体方案、使用方法、测试与基准，以及为适配新版所做的关键改动点（InfiniLM + InfiniCore）。

> 结论（当前能力边界）
> - ✅ 支持「普通 KVCache」的 **in-place 压缩**（压缩后 KV 长度变短，注意力计算 total_len 变小）。
> - ✅ 已验证模型：`/data/huggingface/llava-1.5-7b-hf`、`/data/huggingface/MiniCPM-V-2_6`。
> - ✅ 支持 NVIDIA 后端（本地验证通过）。
> - ❌ 不支持 `PagedKVCache` / paged attention（本方法只能作用在普通 KVCache）。
> - ⚠️ 压缩接口目前要求 `ndev=1`（单卡 / 单 device）。

---

## 1. 术语与接口说明

### 1.1 普通 KVCache vs PagedKVCache

- **普通 KVCache**：连续张量存储，layout 为 `[seq, kv_heads, head_dim]`，由 `createKVCache()` 分配（C API）。
- **PagedKVCache**：用于 paged attention 的 block/page 管理（InfiniLM 新版的 cache 模块），本压缩路径不支持。

### 1.2 压缩后的推理位置（`req_pos` vs `kv_pos`）

压缩会把 KVCache 的「有效长度」从 `ntok` 变为 `new_len`（更短）。为了让解码继续：

- `req_pos`：用于 RoPE / pos_ids（仍然按真实 token 位置递增）
- `kv_pos`：用于 KVCache 写入/读取位置（压缩后从 `new_len` 开始递增）

因此压缩后解码必须使用 `infer_batch_ex` / `infer_batch_ex_with_logits`（可解耦 `req_pos` 与 `kv_pos`），而不是普通的 `infer_batch`。

---

## 2. 构建与安装（NVIDIA）

### 2.1 编译安装 InfiniCore（算子库）

```bash
cd /home/zhujianian/131/InfiniLM2.0/InfiniCore
xmake f -m release --nv-gpu=y
xmake -j$(nproc)
xmake install
```

> 说明：本次适配补齐了 KV 压缩路径需要的算子（如 `quickgelu` / `gelutanh`），并按 NVIDIA 方式编译。

### 2.2 编译安装 InfiniLM（推理框架）

```bash
cd /home/zhujianian/131/InfiniLM2.0/InfiniLM
xmake f -m release
xmake -j$(nproc)
xmake install
```

> 说明：`libinfinicore_infer.so` 已加 `RUNPATH=$ORIGIN`，因此 Python `ctypes.CDLL($INFINI_ROOT/lib/libinfinicore_infer.so)` 不再依赖 `LD_LIBRARY_PATH`。

---

## 3. 权重准备：`.pth` 转 `.bin`

压缩器权重不能直接使用 `.pth`，需要转换成项目支持的二进制格式（无 PyTorch 依赖）。

转换脚本：`scripts/convert_kv_compressor_pth_to_bin.py`

### 3.1 LLaVA 压缩器（本地路径）

```bash
cd /home/zhujianian/131/InfiniLM2.0/InfiniLM
python3 scripts/convert_kv_compressor_pth_to_bin.py \
  --input /home/zhujianian/cvpr/ckpt_store/best_finetune_mlp_1030_mm_9.pth \
  --output ./compress_ckpt/llava_mlp_local.bin \
  --dtype fp16
```

### 3.2 MiniCPM-V 压缩器（本地路径）

```bash
cd /home/zhujianian/131/InfiniLM2.0/InfiniLM
python3 scripts/convert_kv_compressor_pth_to_bin.py \
  --input /home/zhujianian/cvpr/ckpt_store/best_finetune_mlp_13B_mm_1_minicpm.pth \
  --output ./compress_ckpt/minicpmv_mlp_local.bin \
  --dtype fp16
```

二进制格式说明见：`docs/KVCacheCompressionWeightFormat.md`。

---

## 4. 使用方法（真实模型推理）

### 4.1 LLaVA（多模态对话，开启 KV 压缩）

```bash
cd /home/zhujianian/131/InfiniLM2.0/InfiniLM
python3 scripts/llava_chat.py \
  --dev nvidia \
  --ndev 1 \
  --model-dir /data/huggingface/llava-1.5-7b-hf \
  --image /home/zhujianian/cvpr/wuhang/bus.jpg \
  --question "Describe this image." \
  --max-new-tokens 60 \
  --kv-compress \
  --kv-compress-bin ./compress_ckpt/llava_mlp_local.bin \
  --kv-compress-factor 5 \
  --kv-compress-min-seq-len 2 \
  --time
```

说明：
- LLaVA 的 `image_kv_len` 由脚本自动根据 `<image>` token block 推断。
- `--ndev 1` 必须为 1（当前压缩 C API 只支持单 device）。

### 4.2 MiniCPM-V（多模态对话，开启 KV 压缩）

```bash
cd /home/zhujianian/131/InfiniLM2.0/InfiniLM
python3 scripts/minicpmv_chat.py \
  --dev nvidia \
  --model-dir /data/huggingface/MiniCPM-V-2_6 \
  --image /home/zhujianian/cvpr/wuhang/bus.jpg \
  --question "图片是什么？" \
  --max-steps 128 \
  --max-tokens 768 \
  --kv-compress \
  --kv-compress-bin ./compress_ckpt/minicpmv_mlp_local.bin \
  --kv-compress-factor 5 \
  --kv-compress-min-seq-len 2 \
  --kv-compress-image-len 0 \
  --time
```

说明：
- `--kv-compress-image-len 0` 表示自动从 `image_bound` 推断「图像 KV 前缀长度」。
- 如果你只有 **text-only** 压缩权重（没有 `compress_ik/compress_iv`），则将 `--kv-compress-image-len` 设为 `0`/或明确为 0，表示图像前缀不压缩（Hybrid 模式）。

---

## 5. 单测（与 PR 同款）

位置：`tests/`

### 5.1 编译（示例用 g++ 直编）

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

### 5.2 运行

```bash
cd /home/zhujianian/131/InfiniLM2.0/InfiniLM/tests
./test_kv_compression_load ../compress_ckpt/llava_mlp.bin nvidia
./test_kv_compression_correctness ../compress_ckpt/llava_mlp.bin nvidia
./test_kv_compression_correctness_cpu ../compress_ckpt/llava_mlp.bin
```

> `tests/dump_kv/` 中包含测试所需的 `meta.json`、`input_kv.bin`、`output_kv.bin`。

---

## 6. 吞吐测试（对比不开压缩）

注意：这里的 baseline 指 **不启用 KV 压缩**（仍然使用 KV cache），不是“不使用 KV cache”。

### 6.1 语言侧吞吐（LLaVA / MiniCPM-V language-only）

脚本：`scripts/bench_kv_compression_throughput.py`

```bash
cd /home/zhujianian/131/InfiniLM2.0/InfiniLM
python3 scripts/bench_kv_compression_throughput.py \
  --dev nvidia \
  --model both \
  --batch-sizes 1,32,64,128 \
  --prompt-len 640 \
  --decode-steps 64 \
  --warmup-steps 4 \
  --compression-factor 5 \
  --min-seq-len 2 \
  --llava-model-dir /data/huggingface/llava-1.5-7b-hf \
  --llava-compress-bin ./compress_ckpt/llava_mlp_local.bin \
  --llava-max-tokens 768 \
  --llava-image-kv-len 576 \
  --minicpmv-model-dir /data/huggingface/MiniCPM-V-2_6 \
  --minicpmv-compress-bin ./compress_ckpt/minicpmv_mlp_local.bin \
  --minicpmv-max-tokens 768 \
  --minicpmv-image-kv-len 64
```

样例结果（decode TPS）：
- LLaVA：`94.83→96.37`, `327.24→334.17`, `333.92→341.54`, `337.90→344.22`
- MiniCPM-V：`92.86→95.20`, `366.95→368.78`, `376.98→378.61`, `381.65→381.55`

### 6.2 MiniCPM-V 真实图片吞吐（bs=1/8/16/32/64）

脚本：`scripts/bench_minicpmv_mm_kv_compression_throughput.py`

```bash
cd /home/zhujianian/131/InfiniLM2.0/InfiniLM
python3 scripts/bench_minicpmv_mm_kv_compression_throughput.py \
  --dev nvidia \
  --model-dir /data/huggingface/MiniCPM-V-2_6 \
  --image /home/zhujianian/cvpr/wuhang/bus.jpg \
  --question "图片是什么？" \
  --batch-sizes 1,8,16,32,64 \
  --max-tokens 768 \
  --decode-steps 64 \
  --warmup-steps 4 \
  --compress-bin ./compress_ckpt/minicpmv_mlp_local.bin \
  --compression-factor 5 \
  --min-seq-len 2
```

样例结果（decode TPS；prompt_len=519，image_kv_len=492）：
- `bs=1`: `93.83 → 94.35`
- `bs=8`: `319.54 → 322.85`
- `bs=16`: `349.37 → 353.19`
- `bs=32`: `368.67 → 372.20`
- `bs=64`: `376.67 → 380.27`

---

## 7. 关键改动点（0.1.0 → 0.2.0 迁移/适配）

### 7.1 InfiniLM（推理框架）侧

- KV 压缩实现与 C API：
  - `src/cache_manager/kv_compression.hpp`
  - `src/cache_manager/kv_compression.cpp`
  - `src/cache_manager/kv_compression_impl.cpp`
  - `src/cache_manager/kv_compression_capi.cpp`
  - `include/infinicore_infer/kv_compression.h`
- Jiuge 推理接口支持 `req_pos`/`kv_pos` 解耦（压缩后 decode 必需）：
  - `include/infinicore_infer/models/jiuge.h`
  - `src/models/jiuge/jiuge.cpp`
- Python ctypes 绑定（调用 `compressKVCacheInplace` + `infer_batch_ex`）：
  - `scripts/libinfinicore_infer/base.py`
  - `scripts/libinfinicore_infer/jiuge.py`
- LLaVA / MiniCPM-V 脚本接入压缩（prefill 后压缩，decode 用 ex 接口）：
  - `scripts/llava.py`, `scripts/llava_chat.py`
  - `scripts/minicpmv_chat.py`
- 构建侧：为 `libinfinicore_infer.so` 增加 `RUNPATH=$ORIGIN`，避免运行时依赖 `LD_LIBRARY_PATH`：
  - `xmake.lua`
- 稳定性：避免 Python 退出阶段触发 CUDA runtime 卸载问题（压缩器 cache 常驻进程生命周期）：
  - `src/cache_manager/kv_compression_capi.cpp`

### 7.2 InfiniCore（算子库）侧

补齐压缩路径所需算子并适配 NVIDIA 构建（本 workspace 下 `InfiniCore/`）：
- 新增/补齐算子：
  - `InfiniCore/include/infiniop/ops/quickgelu.h`
  - `InfiniCore/src/infiniop/ops/quickgelu/`
  - `InfiniCore/include/infiniop/ops/gelutanh.h`
  - `InfiniCore/src/infiniop/ops/gelutanh/`
- 聚合头更新：`InfiniCore/include/infiniop.h`
- 构建脚本更新：`InfiniCore/xmake/nvidia.lua`（以及 `InfiniCore/xmake/hygon.lua`）

---

## 8. 常见问题

### 8.1 为什么压缩只支持普通 KVCache？

PagedKVCache 的存储不是连续 `[seq, heads, dim]`，而是 block/page 管理；本方法的 in-place 写回与长度收缩假设连续前缀存储，因此无法直接复用。

### 8.2 为什么需要 `ndev=1`？

当前 `compressKVCacheInplace()` 仅支持单 device 的 KVCache（用于快速落地）；多卡/TP 的 KV 压缩需要额外的 per-shard 权重切分、同步策略与跨设备一致的长度维护。

---

## 9. 后续工作（Paged Attention 适配建议）

当前实现只覆盖「普通 KVCache」。如果要支持 `PagedKVCache` / paged attention，可参考以下迁移路径：

1. **Paged KV → 连续 KV（gather）**  
   按 block/page 的 metadata 将有效 token 前缀拼成连续 `[seq, heads, dim]` 张量（每层一份），并转换为压缩器需要的输入 layout。

2. **压缩 + 写回（scatter）**  
   对连续 KV 执行压缩后，将压缩结果按 block/page 顺序写回 paged KV（只覆盖压缩后的前缀）。

3. **更新有效长度与 slot 映射**  
   需要在调度/解码路径引入「有效 KV 长度」（类似 `req_pos` vs `kv_pos` 的解耦思路），保证 decode 的写入位置和 attention 的读取位置使用压缩后的长度。

4. **释放多余 block / page**  
   压缩后可回收多余的 block/page，实现真正的显存回收；同时要维护 block/page 的 ref_count 和 hash 状态（如有）。

5. **多设备支持（可选）**  
   若启用 TP/多卡，需要在各 shard 上独立压缩，并确保跨设备长度一致与同步。

以上工作完成后，压缩器本身无需变化，只需在 paged KV 的读写与长度管理层做适配。
