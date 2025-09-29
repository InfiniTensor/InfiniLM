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
