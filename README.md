# InfiniCore-Infer

本项目是基于 [`InfiniCore`](https://github.com/InfiniTensor/InfiniCore) 的推理引擎。

## 使用方式

- 编译并安装 `InfiniCore` 。注意根据提示设置好 `INFINI_ROOT` 环境变量（默认为 `$HOME/.infini`）。

- 编译并安装 `InfiniCore-Infer`

```bash
xmake && xmake install
```

- 运行模型推理测试

```bash
python jiuge.py [--cpu | --nvidia | --cambricon | --ascend | --metax | --moore] <path/to/model_dir> [n_device]
```

- 部署模型推理服务

```bash
launch_server.py [-h] [--dev {cpu,nvidia,cambricon,ascend,metax,moore}]
                        [--model-path MODEL_PATH] [--ndev NDEV] [--max-batch MAX_BATCH]
                        [--max-tokens MAX_TOKENS]
```
