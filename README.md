# InfiniLM

![star](https://atomgit.com/InfiniTensor/InfiniLM/star/badge.svg)

本项目是基于 [`InfiniCore`](https://github.com/InfiniTensor/InfiniCore) 的推理引擎。

当前版本依赖[`InfiniCore v0.2.9`](https://github.com/InfiniTensor/InfiniCore/releases/tag/v0.2.9)版本。

### Qwen built-in MTP (experimental)

The text-only greedy path supports one built-in MTP layer, 1–4 draft tokens,
paged KV caching and PP1. Target verification selects matching Conv/GDN
checkpoints without replay. The scheduler batches target verification while
keeping acceptance and state independent for each request. TP1 also batches
draft continuation; TP2 retains separate draft calls because its end-to-end
batching benefit has not been established.
It handles cancellation, EOS/output limits and ordinary Decode when speculative
cache capacity is unavailable. This requires the matching FP8/MTP InfiniCore build;
the release dependency listed above does not contain these additions. Track the
runtime patch in [InfiniCore #1565](https://github.com/InfiniTensor/InfiniCore/issues/1565).
Graph execution additionally requires the graph lifetime/recording fixes in
[InfiniCore #1560](https://github.com/InfiniTensor/InfiniCore/pull/1560).

For Qwen3.8-27B-FP8 with E4M3 weights and 128×128 weight blocks, set
`quantization_config.fp8_backend` to `"marlin"` in the checkpoint configuration
for NVIDIA inference. Weights are packed once during loading using the existing
Marlin operator; a separately converted weight file is unnecessary. The default
`"compatibility"` backend dequantizes on the device at execution time and is slower.

Example on one A6000 with the Marlin configuration:

```bash
python -m infinilm.server.inference_server \
  --model /models/Qwen3.8-27B-FP8-marlin --device nvidia --dtype bfloat16 \
  --enable-paged-attn --enable-mtp --num-draft-tokens 2 \
  --max-batch-size 2 --num-state-rows 9 --num-blocks 40 --block-size 64 \
  --disable-prefix-caching --top-k 1 --max-new-tokens 64
```

The same MTP and cache options are accepted by `examples/test_infer.py`,
`examples/bench.py` and `test/bench/test_benchmark.py`. The offline benchmark
reuses the scheduler-backed `LLM` path for MTP; it does not time model loading.

`num_state_rows` counts the zero row, committed request states and speculative
checkpoints. Its MTP default is `1 + max_batch_size * (num_draft_tokens + 2)`,
independent of KV page count. A smaller pool can reduce concurrent admission or
use ordinary Decode when checkpoint rows are unavailable. The page budget must
also accommodate each prompt and its requested output limit.

For exact full-prompt reuse, replace `--disable-prefix-caching` with
`--mtp-prefix-cache-mib 512`. This TP1-only LRU cache owns device copies of both
target/draft KV, recurrent state and the initial MTP outputs. Hits restore into
request-owned pages and state rows. The budget limits live snapshot tensor
storage, not model memory, allocator reservations or total process memory.
Partial-prefix matching and the Attention-only cache's SLRU policy are not
supported by this hybrid snapshot cache. Cache reset or weight loading
invalidates snapshots.

`--enable-graph` currently requires `--num-draft-tokens 1 --max-batch-size 1`.
It captures ordinary Decode and supported short draft shapes; Prefill and target
verification remain eager. Multi-candidate and batched execution use eager.
Random sampling, multimodal requests, multi-layer MTP service execution and remote
state transfer are rejected. NVIDIA A6000 validation covers TP1 and TP2 greedy
execution, batched requests, cancellation and cache reclamation. The 27B FP8 TP2
checks use K=2; K=1/2/4 and graph recapture are additionally checked with a tiny
checkpoint. Exact full-prompt caching remains TP1-only. Other accelerators have
not been validated for this service path.

Control-flow and GPU integration checks:

```bash
python -m pytest test/models/qwen3_5 -q
INFINILM_QWEN_MTP_TEST_MODEL=/models/tiny-qwen-mtp \
INFINILM_QWEN_MTP_TEST_TP=1 python -m pytest \
  test/models/qwen3_5 -q
```

For the TP2 execution and batching checks, expose two GPUs and set
`INFINILM_QWEN_MTP_TEST_TP=2`; the prefix-cache check still uses TP1.
The three test modules cover CPU scheduling/lifecycle, GPU execution, and
checkpoint/model contracts. GPU checks skip when no test checkpoint is set.

## 使用方式
#### 一、编译并安装 `InfiniCore`
编译并安装 `InfiniCore`， 详情见 InfiniCore的 [`README`](https://github.com/InfiniTensor/InfiniCore) :

- 注意根据提示设置好 `INFINI_ROOT` 环境变量（默认为 `$HOME/.infini`）
- 根据硬件平台，选择 xmake 构建配置
- 编译安装InfiniCore
- 安装 C++ 库
- 安装 Python 包


#### 二、编译并安装  `InfiniLM`
  - 克隆项目

    由于仓库中含有子模块，所以在克隆时请添加 `--recursive` 或 `--recurse-submodules`，如：

    ```shell
    git clone --recursive https://github.com/InfiniTensor/InfiniLM.git
    ```

    或者在普通克隆后进行更新：

    ```shell
    git submodule update --init --recursive
    ```

  - 安装 InfiniLM Python 包
    ```bash
      pip install -e .
    ```

    使用 CoreX PyTorch 时，需要让 InfiniLM 与 PyTorch/InfiniCore
    使用相同的 libstdc++ ABI：

    ```bash
    export INFINILM_CXX11_ABI=0
    pip install -e .
    ```

  - 单次推理测试
    - llama示例
    ```bash
    python examples/test_infer.py --device [cpu | nvidia | qy | metax | moore | iluvatar | ali | cambricon | hygon] --model=<path/to/model_dir>
    ```
    - 例如：
    ```bash
    python examples/test_infer.py --device=nvidia --model=/models/TinyLlama-1.1B-Chat-v1.0
    ```
  - 分布式推理测试
      - 9g示例
      ```bash
    python examples/test_infer.py [-- device nvidia] --model=<path/to/model> --backend=cpp --tp=NDEV --batch-size=MAX_BATCH
    ```

    - 例如： 9G7B模型，cpp后端，batch_size为16，4卡分布式
    ```bash
    python examples/test_infer.py --device nvidia --model=/models/9G7B_MHA/ --backend=cpp --tp=4 --batch-size=16
    ```

    - PP=2 示例：

      在两个终端中分别启动 stage 0 和 stage 1。两个进程的模型、并行和缓存参数必须保持一致。

      ```bash
      # Terminal 1: stage 0 / coordinator (--node-rank=0)
      CUDA_VISIBLE_DEVICES=0 python examples/test_infer.py --device=nvidia --model=<path/to/model> --tp=1 --pp=2 --node-rank=0 --master-addr=127.0.0.1 --master-port=29500 --enable-paged-attn --attn=flash-attn --num-blocks=128

      # Terminal 2: stage 1 / worker
      CUDA_VISIBLE_DEVICES=1 python examples/test_infer.py --device=nvidia --model=<path/to/model> --tp=1 --pp=2 --node-rank=1 --master-addr=127.0.0.1 --master-port=29500 --enable-paged-attn --attn=flash-attn --num-blocks=128
      ```

      跨节点运行时，每个节点的命令中的 `--master-addr` 和 `--master-port` 设置为 stage 0 节点的 IP 地址和通信端口。


  - 推理服务测试
    - 启动推理服务
      ```bash
      python python/infinilm/server/inference_server.py --device [cpu | nvidia | qy | metax | moore | iluvatar | ali | cambricon | hygon] --model=<path/to/model-dir> --max-new-tokens=MAX_TOKENS --max-batch-size=MAX_BATCH --tp=NDEV --temperature=TEMP --top-p=TOP_P --top-k=TOP_K --host=HOST --port=PORT
      ```
    
    - 单卡示例：
      ```bash
      CUDA_VISIBLE_DEVICES=0 python python/infinilm/server/inference_server.py --device nvidia --model=/models/9G7B_MHA/ --max-new-tokens=100 --max-batch-size=32 --tp=1 --temperature=1.0 --top-p=0.8 --top-k=1
      ```
    
    - 多卡分布式示例：
      ```bash
      CUDA_VISIBLE_DEVICES=0,1,2,3 python python/infinilm/server/inference_server.py --device nvidia --model=/models/9G7B_MHA/ --max-new-tokens=100 --max-batch-size=32 --tp=4 --temperature=1.0 --top-p=0.8 --top-k=1
      ```
    
    - 使用paged attention, flash attention后端，cuda graph等功能：
      ```bash
      CUDA_VISIBLE_DEVICES=0,1,2,3 python python/infinilm/server/inference_server.py --device nvidia --model=/models/9G7B_MHA/ --enable-paged-attn --attn=flash-attn --enable-graph
      ```

    - PP=2 推理服务示例：

      只有 stage 0 启动 HTTP 服务。两个进程使用相同的 PP rendezvous 地址和模型配置。

      ```bash
      # Terminal 1: stage 0 / coordinator and HTTP server
      python python/infinilm/server/inference_server.py --device=nvidia --model=<path/to/model> --tp=1 --pp=2 --node-rank=0 --master-addr=<HOST.IP> --master-port=29500 --enable-paged-attn --attn=flash-attn --num-blocks=128 --max-batch-size=32 --port=8000

      # Terminal 2: stage 1 / worker
      python python/infinilm/server/inference_server.py --device=nvidia --model=<path/to/model> --tp=1 --pp=2 --node-rank=1 --master-addr=<HOST.IP> --master-port=29500 --enable-paged-attn --attn=flash-attn --num-blocks=128 --max-batch-size=32 --port=8000
      ```
    
    - 测试推理服务性能：
      ```bash
      python scripts/test_perf.py --verbose
      ```

    - 单请求推理服务测试
      ```bash
      python test/service/request.py --content="text:Image 1:" --content="image_url:xxx.jpg" --content="text:Image 2:" --content="image_url:xxxx.jpg" --content="text:Compare the 2 images."
      ```

  - 运行推理基准测试（C-Eval/MMLU）

    ```bash
    python test/bench/test_benchmark.py --device [cpu | nvidia | qy | metax | moore | iluvatar | ali | cambricon | hygon] --model <path/to/model_dir> --bench {ceval|mmlu} [--backend cpp] [--tp N] [--subject SUBJECT] [--num-samples N] [--max-new-tokens N] [--output-csv PATH] [--cache-dir PATH]
    ```

    - 参数说明：
      - `--subject`: 指定科目，支持单个科目、多个科目（逗号分隔）或 `all`（默认值，加载全部科目）
      - `--output-csv`: 可选，指定CSV输出文件路径。如未指定则不生成CSV文件。CSV包含每个科目的结果和总体结果
      - `--cache-dir`: 可选，指定数据集缓存目录的父目录。应指向包含 `ceval___ceval-exam` 和 `cais___mmlu` 等数据集子目录的父目录（例如 `~/.cache/huggingface/datasets/`）。设置后脚本优先使用本地 CSV（`pandas.read_csv`）离线加载数据，避免 `load_dataset` 的网络请求

    - C-Eval示例：
      - 单个科目：
        ```bash
        python test/bench/test_benchmark.py --device nvidia /models/9G7B_MHA --bench ceval --subject middle_school_mathematics --num-samples 100 --backend cpp --tp 1
        ```
      - 多个科目（逗号分隔）：
        ```bash
        python test/bench/test_benchmark.py --device nvidia /models/9G7B_MHA --bench ceval --subject middle_school_mathematics,high_school_physics --backend cpp --tp 1 --output-csv results.csv
        ```
      - 全部科目并输出CSV：
        ```bash
        python test/bench/test_benchmark.py --device nvidia /models/9G7B_MHA --bench ceval --subject all --backend cpp --tp 1 --output-csv results.csv
        ```
      - 使用缓存目录加速加载：
        ```bash
        python test/bench/test_benchmark.py --device nvidia /models/9G7B_MHA --bench ceval --subject middle_school_mathematics --backend cpp --tp 1 --cache-dir ~/.cache/huggingface/datasets/
        ```
        > 注意：`--cache-dir` 应指向包含 `ceval___ceval-exam` 和 `cais___mmlu` 等数据集子目录的父目录，而不是直接指向这些子目录

    - MMLU示例：
      - 单个科目：
        ```bash
        python test/bench/test_benchmark.py --device nvidia /models/9G7B_MHA --bench mmlu --subject abstract_algebra --backend cpp --tp 1
        ```
      - 多个科目（逗号分隔）：
        ```bash
        python test/bench/test_benchmark.py --device nvidia /models/9G7B_MHA --bench mmlu --subject abstract_algebra,anatomy,astronomy --backend cpp --tp 1 --output-csv results.csv
        ```
      - 使用缓存目录加速加载：
        ```bash
        python test/bench/test_benchmark.py --device nvidia /models/9G7B_MHA --bench mmlu --subject abstract_algebra --backend cpp --tp 1 --cache-dir ~/.cache/huggingface/datasets/
        ```
        > 注意：`--cache-dir` 应指向包含 `ceval___ceval-exam` 和 `cais___mmlu` 等数据集子目录的父目录，而不是直接指向这些子目录

  - 试验中功能
    - Warm Up
      ```bash
      python examples/bench.py --device nvidia --model=<model-path> --warmup
      ```
    - Paged Attention
      ```bash
      python examples/bench.py --device nvidia --model=<model-path> --enable-paged-attn
      ```
    - CUDA Graph
      ```bash
      python examples/bench.py --device nvidia --model=<model-path> --enable-paged-attn --enable-graph
      ```
    - 选择attention后端 (使用flash attention后端需要先在InfiniCore完成相关配置和编译)
      ```bash
      python examples/bench.py --device nvidia --model=<model-path> --enable-paged-attn [--attn=default | --attn=flash-attn]
      ```
