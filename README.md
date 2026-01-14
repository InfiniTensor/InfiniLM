# InfiniLM

本项目是基于 [`InfiniCore`](https://github.com/InfiniTensor/InfiniCore) 的推理引擎。

## 使用方式

- 编译并安装 `InfiniCore` 。注意根据提示设置好 `INFINI_ROOT` 环境变量（默认为 `$HOME/.infini`）。

- 编译并安装 `InfiniLM`

```bash
xmake && xmake install
```

- 运行模型推理测试

```bash
python scripts/jiuge.py [--cpu | --nvidia | --qy | --cambricon | --ascend | --metax | --moore | --iluvatar | --kunlun | --hygon] path/to/model_dir [n_device]
```

- 部署模型推理服务

```bash
python scripts/launch_server.py --model-path MODEL_PATH [-h] [--dev {cpu,nvidia,qy, cambricon,ascend,metax,moore,iluvatar,kunlun,hygon}] [--ndev NDEV] [--max-batch MAX_BATCH] [--max-tokens MAX_TOKENS]
```

- 测试模型推理服务性能

```bash
python scripts/test_perf.py
```

- 使用推理服务测试模型困惑度（Perplexity）

```bash
python scripts/test_ppl.py --model-path MODEL_PATH [--ndev NDEV] [--max-batch MAX_BATCH] [--max-tokens MAX_TOKENS]
```

## 使用方式(新版)
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

  - 单次推理测试
    - llama示例
    ```bash
    python examples/llama.py [--cpu | --nvidia | --metax | --moore | --iluvatar] --model_path=<path/to/model_dir>
    ```
    - 例如：
    ```bash
    python examples/llama.py --nvidia --model_path=/models/TinyLlama-1.1B-Chat-v1.0
    ```
  - 分布式推理测试
      - 9g示例
      ```bash
    python examples/jiuge.py [---nvidia] --model_path=<path/to/model_dir> --backend=cpp --tp=NDEV --batch_size=MAX_BATCH
    ```

    - 例如： 9G7B模型，cpp后端，batch_size为16，4卡分布式
    ```bash
    python examples/jiuge.py --nvidia --model_path=/models/9G7B_MHA/ --backend=cpp --tp=4 --batch_size=16
    ```


  - 推理服务测试
    - 启动推理服务
      ```bash
      python python/infinilm/server/inference_server.py [--cpu | --nvidia | --metax | --moore | --iluvatar | --cambricon] --model_path=<path/to/model_dir> --max_tokens=MAX_TOKENS --max_batch_size=MAX_BATCH --tp=NDEV --temperature=TEMP --top_p=TOP_P --top_k=TOP_K --host=HOST --port=PORT
      ```
    
    - 单卡示例：
      ```bash
      CUDA_VISIBLE_DEVICES=0 python python/infinilm/server/inference_server.py --nvidia --model_path=/models/9G7B_MHA/ --max_tokens=100 --max_batch_size=32 --tp=1 --temperature=1.0 --top_p=0.8 --top_k=1
      ```
    
    - 多卡分布式示例：
      ```bash
      CUDA_VISIBLE_DEVICES=0,1,2,3 python python/infinilm/server/inference_server.py --nvidia --model_path=/models/9G7B_MHA/ --max_tokens=100 --max_batch_size=32 --tp=4 --temperature=1.0 --top_p=0.8 --top_k=1
      ```
    
    - 测试推理服务性能：
      ```bash
      python scripts/test_perf.py --verbose
      ```

  - 运行推理基准测试（C-Eval/MMLU）

    ```bash
    python test/bench/test_benchmark.py [--cpu | --nvidia | --cambricon | --ascend | --metax | --moore | --iluvatar | --kunlun | --hygon] <path/to/model_dir> --bench {ceval|mmlu} [--backend cpp] [--ndev N] [--subject SUBJECT] [--num_samples N] [--max_new_tokens N] [--output_csv PATH] [--cache_dir PATH]
    ```

    - 参数说明：
      - `--subject`: 指定科目，支持单个科目、多个科目（逗号分隔）或 `all`（默认值，加载全部科目）
      - `--output_csv`: 可选，指定CSV输出文件路径。如未指定则不生成CSV文件。CSV包含每个科目的结果和总体结果
      - `--cache_dir`: 可选，指定数据集缓存目录的父目录。应指向包含 `ceval___ceval-exam` 和 `cais___mmlu` 等数据集子目录的父目录（例如 `~/.cache/huggingface/datasets/`）。设置后脚本优先使用本地 CSV（`pandas.read_csv`）离线加载数据，避免 `load_dataset` 的网络请求

    - C-Eval示例：
      - 单个科目：
        ```bash
        python test/bench/test_benchmark.py --nvidia /models/9G7B_MHA --bench ceval --subject middle_school_mathematics --num_samples 100 --backend cpp --ndev 1
        ```
      - 多个科目（逗号分隔）：
        ```bash
        python test/bench/test_benchmark.py --nvidia /models/9G7B_MHA --bench ceval --subject middle_school_mathematics,high_school_physics --backend cpp --ndev 1 --output_csv results.csv
        ```
      - 全部科目并输出CSV：
        ```bash
        python test/bench/test_benchmark.py --nvidia /models/9G7B_MHA --bench ceval --subject all --backend cpp --ndev 1 --output_csv results.csv
        ```
      - 使用缓存目录加速加载：
        ```bash
        python test/bench/test_benchmark.py --nvidia /models/9G7B_MHA --bench ceval --subject middle_school_mathematics --backend cpp --ndev 1 --cache_dir ~/.cache/huggingface/datasets/
        ```
        > 注意：`--cache_dir` 应指向包含 `ceval___ceval-exam` 和 `cais___mmlu` 等数据集子目录的父目录，而不是直接指向这些子目录

    - MMLU示例：
      - 单个科目：
        ```bash
        python test/bench/test_benchmark.py --nvidia /models/9G7B_MHA --bench mmlu --subject abstract_algebra --backend cpp --ndev 1
        ```
      - 多个科目（逗号分隔）：
        ```bash
        python test/bench/test_benchmark.py --nvidia /models/9G7B_MHA --bench mmlu --subject abstract_algebra,anatomy,astronomy --backend cpp --ndev 1 --output_csv results.csv
        ```
      - 使用缓存目录加速加载：
        ```bash
        python test/bench/test_benchmark.py --nvidia /models/9G7B_MHA --bench mmlu --subject abstract_algebra --backend cpp --ndev 1 --cache_dir ~/.cache/huggingface/datasets/
        ```
        > 注意：`--cache_dir` 应指向包含 `ceval___ceval-exam` 和 `cais___mmlu` 等数据集子目录的父目录，而不是直接指向这些子目录
