# InfiniCore

[![Doc](https://img.shields.io/badge/Document-ready-blue)](https://github.com/InfiniTensor/InfiniCore-Documentation)
[![CI](https://github.com/InfiniTensor/InfiniCore/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/InfiniTensor/InfiniCore/actions)
[![license](https://img.shields.io/github/license/InfiniTensor/InfiniCore)](https://mit-license.org/)
![GitHub repo size](https://img.shields.io/github/repo-size/InfiniTensor/InfiniCore)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/InfiniTensor/InfiniCore)

[![GitHub Issues](https://img.shields.io/github/issues/InfiniTensor/InfiniCore)](https://github.com/InfiniTensor/InfiniCore/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/InfiniTensor/InfiniCore)](https://github.com/InfiniTensor/InfiniCore/pulls)
![GitHub contributors](https://img.shields.io/github/contributors/InfiniTensor/InfiniCore)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/InfiniTensor/InfiniCore)

InfiniCore 是一个跨平台统一编程工具集，为不同芯片平台的功能（包括计算、运行时、通信等）提供统一 C 语言接口。目前支持的硬件和后端包括：

- CPU；
- CUDA
  - 英伟达 GPU；
  - 摩尔线程 GPU；
  - 天数智芯 GPU；
  - 沐曦 GPU；
  - 曙光 DCU；
- 华为昇腾 NPU；
- 寒武纪 MLU；
- 昆仑芯 XPU；

## 配置和使用

### 一键安装

在 `script/` 目录中提供了 `install.py` 安装脚本。使用方式如下：

```shell
cd InfiniCore

python scripts/install.py [XMAKE_CONFIG_FLAGS]
```

参数 `XMAKE_CONFIG_FLAGS` 是 xmake 构建配置，可配置下列可选项：

| 选项                     | 功能                          | 默认值
|--------------------------|-------------------------------|:-:
| `--omp=[y\|n]`           | 是否使用 OpenMP               | y
| `--cpu=[y\|n]`           | 是否编译 CPU 接口实现         | y
| `--nv-gpu=[y\|n]`        | 是否编译英伟达 GPU 接口实现   | n
| `--ascend-npu=[y\|n]`    | 是否编译昇腾 NPU 接口实现     | n
| `--cambricon-mlu=[y\|n]` | 是否编译寒武纪 MLU 接口实现   | n
| `--metax-gpu=[y\|n]`     | 是否编译沐曦 GPU 接口实现     | n
| `--moore-gpu=[y\|n]`     | 是否编译摩尔线程 GPU 接口实现 | n
| `--sugon-dcu=[y\|n]`     | 是否编译曙光 DCU 接口实现     | n
| `--kunlun-xpu=[y\|n]`    | 是否编译昆仑 XPU 接口实现     | n
| `--ccl=[y\|n]`           | 是否编译 InfiniCCL 通信库接口实现     | n

### 手动安装

1. 项目配置

   - 查看当前配置

     ```shell
     xmake f -v
     ```

   - 配置 CPU（默认配置）

     ```shell
     xmake f -cv
     ```

   - 配置加速卡

     ```shell
     # 英伟达
     # 可以指定 CUDA 路径， 一般环境变量为 `CUDA_HOME` 或者 `CUDA_ROOT`
     xmake f --nv-gpu=true --cuda=$CUDA_HOME -cv

     # 寒武纪
     xmake f --cambricon-mlu=true -cv

     # 华为昇腾
     xmake f --ascend-npu=true -cv
     ```

2. 编译安装

   默认安装路径为 `$HOME/.infini`。

   ```shell
   xmake build && xmake install
   ```

3. 设置环境变量

   按输出提示设置 `INFINI_ROOT` 和 `LD_LIBRARY_PATH` 环境变量。

### 运行测试

#### 运行Python算子测试

```shell
python test/infiniop/[operator].py [--cpu | --nvidia | --cambricon | --ascend]
```

#### 一键运行所有Python算子测试

```shell
python scripts/python_test.py [--cpu | --nvidia | --cambricon | --ascend]
```

#### 算子测试框架

详见 `test/infiniop-test` 目录

#### 通信库（InfiniCCL）测试

编译（需要先安装InfiniCCL）：
```shell
xmake build infiniccl-test
```

在英伟达平台运行测试（会自动使用所有可见的卡）：
```shell
infiniccl-test --nvidia
```

## 开发指南

### 代码格式化

本项目使用 [`scripts/format.py`](/scripts/format.py) 脚本实现代码格式化检查和操作。

使用

```shell
python scripts/format.py -h
```

查看脚本帮助信息：

```plaintext
usage: format.py [-h] [--ref REF] [--path [PATH ...]] [--check] [--c C] [--py PY]

options:
  -h, --help         show this help message and exit
  --ref REF          Git reference (commit hash) to compare against.
  --path [PATH ...]  Files to format or check.
  --check            Check files without modifying them.
  --c C              C formatter (default: clang-format-16)
  --py PY            Python formatter (default: black)
```

参数中：

- `ref` 和 `path` 控制格式化的文件范围
  - 若 `ref` 和 `path` 都为空，格式化当前暂存（git added）的文件；
  - 否则
    - 若 `ref` 非空，将比较指定 commit 和当前代码的差异，只格式化修改过的文件；
    - 若 `path` 非空，可传入多个路径（`--path p0 p1 p2`），只格式化指定路径及其子目录中的文件；
- 若设置 `--check`，将检查代码是否需要修改格式，不修改文件内容；
- 通过 `--c` 指定 c/c++ 格式化器，默认为 `clang-format-16`；
- 通过 `--python` 指定 python 格式化器 `black`；

### vscode 开发配置

基本配置见 [xmake 官方文档](https://xmake.io/#/zh-cn/plugin/more_plugins?id=%e9%85%8d%e7%bd%ae-intellsence)。

- TL;DR
  - clangd

    打开 *xmake.lua*，保存一次以触发编译命令生成，将在工作路径下自动生成 *.vscode/compile_commands.json* 文件。然后在这个文件夹下创建 *settings.json*，填入：

    > .vscode/settings.json

    ```json
    {
        "clangd.arguments": [
            "--compile-commands-dir=.vscode"
        ]
    }
    ```
