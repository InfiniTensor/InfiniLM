# InfiniCore

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

4. 运行算子测试

   ```shell
   python test/infiniop/[operator].py [--cpu | --nvidia | --cambricon | --ascend]
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
