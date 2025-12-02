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
python scripts/jiuge.py [--cpu | --nvidia | --cambricon | --ascend | --metax | --moore | --iluvatar | --kunlun | --hygon] path/to/model_dir [n_device]
```

- 部署模型推理服务

```bash
python scripts/launch_server.py --model-path MODEL_PATH [-h] [--dev {cpu,nvidia,cambricon,ascend,metax,moore,iluvatar,kunlun,hygon}] [--ndev NDEV] [--max-batch MAX_BATCH] [--max-tokens MAX_TOKENS]
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

- 编译并安装 `InfiniCore`， 详情见 InfiniCore的 [`README`](https://github.com/InfiniTensor/InfiniCore) :

    - 注意根据提示设置好 `INFINI_ROOT` 环境变量（默认为 `$HOME/.infini`）
    - 根据硬件平台，选择 xmake 构建配置
    - 编译安装InfiniCore
    - 安装 C++ 库
    - 安装 Python 包



- 编译并安装 `InfiniLM` Python 包
  - 安装第三方依赖
  ```bash
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
