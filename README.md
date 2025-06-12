# llama.cu

[![CI](https://github.com/YdrMaster/llama.cu/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/YdrMaster/llama.cu/actions)
[![license](https://img.shields.io/github/license/YdrMaster/llama.cu)](https://mit-license.org/)

[![GitHub Issues](https://img.shields.io/github/issues/YdrMaster/llama.cu)](https://github.com/YdrMaster/llama.cu/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/YdrMaster/llama.cu)](https://github.com/YdrMaster/llama.cu/pulls)
![GitHub repo size](https://img.shields.io/github/repo-size/YdrMaster/llama.cu)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/YdrMaster/llama.cu)
![GitHub contributors](https://img.shields.io/github/contributors/YdrMaster/llama.cu)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/YdrMaster/llama.cu)

## 使用说明

### 帮助信息

```shell
cargo xtask help
```

```plaintext
Usage: xtask <COMMAND>

Commands:
  generate  text generation
  chat      chat in console
  service   web service
  help      Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```

### 文本生成

```shell
cargo generate --help
```

简写：

```shell
cargo gen --help
```

```plaintext
text generation

Usage: xtask generate [OPTIONS] <MODEL>

Arguments:
  <MODEL>

Options:
      --gpus <GPUS>
      --max-steps <MAX_STEPS>
  -p, --prompt <PROMPT>
  -t, --use-template
  -h, --help
```

### 对话

```shell
cargo chat --help
```

```plaintext
chat in console

Usage: xtask chat [OPTIONS] <MODEL>

Arguments:
  <MODEL>

Options:
      --gpus <GPUS>
      --max-steps <MAX_STEPS>
  -h, --help                   Print help
```

### web 服务

```shell
cargo service --help
```

```plaintext
web service

Usage: xtask service [OPTIONS] --port <PORT> <FILE>

Arguments:
  <FILE>

Options:
  -p, --port <PORT>
      --no-cuda-graph
      --name <NAME>
      --gpus <GPUS>
      --max-steps <MAX_STEPS>
      --think
  -h, --help
```

通过 TOML 配置文件可以配置多模型服务。示例格式：

```toml
[model-name]
path = "model-path"
think = true
max-steps = 2048
```
