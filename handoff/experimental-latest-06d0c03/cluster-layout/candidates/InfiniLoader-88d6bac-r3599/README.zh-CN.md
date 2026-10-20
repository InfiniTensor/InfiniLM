# InfiniLoader 中文说明

[English README](README.md)

开发分支新增了模型身份校验、daemon 启动互斥、memfd 写封印、不可变磁盘缓存版本、
容器内存预算及租约保护。新增 `doctor`、`verify`、`run`、`gc` 命令，详见
[运行管理与迁移说明](docs/operations.md)。

```bash
iloader doctor
iloader start "$MODEL" --name dev --local-cache \
  --max-memory-bytes 32GiB --reserve-bytes 4GiB --max-resident-bytes 64GiB
iloader run dev -- vllm serve '{model}'
iloader gc --resident --max-bytes 32GiB       # 只预览
iloader gc --resident --max-bytes 32GiB --apply
```

内存紧张时，可只用 `iloader cache` 构建本地磁盘缓存，再用
`iloader run --cache-path PATH -- COMMAND` 启动框架。原有直接传 view 路径的方式仍可使用；
自动清理无法识别未注册的读者，启用闲置过期或执行清理前应通过 `run` 保护任务。

`InfiniLoader` 是一个面向大模型推理框架开发调试的用户态权重加载管理器。它解决的核心问题是：调试 vLLM、SGLang、InfiniLM 或自研推理框架时，经常需要反复停止、修改代码、重新启动框架，而每次启动都要重新从存储读取模型权重。

默认模式下，InfiniLoader 会把模型目录常驻在内存里，并暴露出一个看起来像普通 Hugging Face 模型目录的路径。框架仍然使用原来的文件读取逻辑，通常只需要把模型路径换成 `iloader start/view` 输出的路径。

## 快速开始

最基础路径是 memfd 支撑的 memfs 风格内存 view。如果你原本会把框架直接指向
`/path/to/huggingface-model`，现在只需要在启动框架前多执行一行命令，然后把模型路径换成它输出的 view。

```bash
cd InfiniLoader
source env.sh

MODEL=/path/to/huggingface-model
MODEL_VIEW=$(iloader start "$MODEL" --name dev-model)
```

之后把 `"$MODEL_VIEW"` 当成模型路径传给推理引擎。`iloader start` 是一样的，只有下面的引擎启动命令不同。

```bash
# InfiniLM
python python/infinilm/server/inference_server.py \
  --device cuda \
  --model="$MODEL_VIEW" \
  --tp=1
```

```bash
# vLLM
vllm serve "$MODEL_VIEW" --tensor-parallel-size 1
```

```bash
# SGLang
python -m sglang.launch_server \
  --model-path "$MODEL_VIEW" \
  --tp 1
```

结束时：

```bash
iloader stop dev-model --remove-view
```

这就是 0.1.0 的稳定兼容路径：不改框架权重加载代码，只换模型路径。vLLM、SGLang、InfiniLM、本地缓存、分布式 view、进度/status 和自动补全放在后面的详细教程里。

## 解决的问题

大模型推理框架调试中，权重加载常常成为启动慢的主要来源：

- 远端或共享存储在读取大 safetensors shard 时容易慢或抖动；
- OS page cache 能改善热启动，但不能消除框架解析、tensor 构建和 CPU 到 GPU 拷贝开销；
- 分布式调试时，多 rank 可能重复扫描或读取同一个 checkpoint；
- 调试框架代码时，不希望为了加速加载而频繁修改框架 loader 接口。

InfiniLoader 的设计目标是让权重常驻服务和框架进程解耦。你可以反复停止和重启 vLLM/SGLang/InfiniLM，权重 view 仍然保持 warm。

## 工作原理

默认 `start/view` 路径只依赖 Linux 用户态能力，不需要 root，也不需要 FUSE：

1. 后台 daemon 读取模型目录中选中的文件。
2. 每个文件被复制到 `memfd_create` 创建的匿名内存文件中。
3. daemon 持有这些文件描述符，使内容保持可读。
4. view 目录镜像原始模型树，文件是指向 `/proc/<daemon-pid>/fd/<fd>` 的符号链接。
5. 推理框架继续使用普通 `open`、`stat`、`mmap` 读取模型文件。

此外，InfiniLoader 还支持本地二级缓存、分布式 rank view 和 TP 切分分析。

## 环境设置和运行目录

`env.sh` 会把仓库的 `src` 目录加入 `PYTHONPATH`，并在当前终端中定义
`iloader` shell function；开发态不再依赖 `bin` 目录下的包装脚本。在 bash 中，
它也会自动注册 iloader 子命令、选项和活跃 cache 名称的补全。

也可以作为 Python 包安装：

```bash
pip install -e .
iloader selftest
```

运行单元测试：

```bash
pip install -e .[test]
python -m pytest
```

运行状态会优先放在 `$ILOADER_BASE_DIR`；未设置时使用
`$XDG_CACHE_HOME/infiniloader` 或 `~/.cache/infiniloader`。所有会创建运行状态的
命令也都支持通过 `--base-dir` 显式指定目录。

## 命令边界

0.1.0 的公开接口刻意保持较小：

- 主流程：`start`、`list`、`status`、`view`、`stop`。
- 工具命令：`inspect`、`bench`、`completion`、`selftest`。
- 稳定高级能力：`cache`、`cache-status`、`cache-list`、`dist-views`、`dist-simulate`、`dist-plan`。

`serve` 是 `start` 内部使用的 daemon 入口，不出现在 help 和自动补全里。透明同路径拦截、全局 mount 管理、GPU resident tensor cache 不属于 0.1.0 CLI；当前稳定契约是把 `iloader start` 输出的 view 路径传给推理引擎。

## 详细使用

### 基础 memfd 缓存

默认 `start` 命令会把模型文件预加载到匿名内存文件，并把 Hugging Face 兼容的 view 路径输出到 stdout。加载期间的进度、吞吐和 ETA 会输出到 stderr。

```bash
MODEL_VIEW=$(iloader start "$MODEL" --name dev-model)
```

任何接受模型目录的框架命令都可以使用这个 view：

```bash
vllm serve "$MODEL_VIEW" --tensor-parallel-size 1
```

```bash
python -m sglang.launch_server \
  --model-path "$MODEL_VIEW" \
  --tp 1
```

```bash
python python/infinilm/server/inference_server.py \
  --device cuda \
  --model="$MODEL_VIEW" \
  --tp=1
```

另一个终端里可以查看和管理当前活跃的 cache：

```bash
iloader list
iloader status --name dev-model
iloader stop dev-model --remove-view
```

如果 cache 仍处于 `starting` 或 `loading`，`stop` 会取消这次 preload。

### 框架反复重启，权重保持 warm

```bash
MODEL_VIEW=$(iloader start "$MODEL" --name dev-model)

# 修改框架代码后反复运行这一条。
vllm serve "$MODEL_VIEW" --tensor-parallel-size 1
```

停止 `vllm serve` 不会停止 InfiniLoader daemon。下一次框架启动会复用同一个 warm view。如果 daemon 已经在运行，`iloader start "$MODEL" --name dev-model` 会直接输出已有 view 路径。

### 先落本地缓存

当原始模型目录位于远端或共享存储，而你希望先构建一个可复用的本地缓存时，建议开启
`--local-cache`：

```bash
iloader cache "$MODEL" --name dev-model --cache-workers 8
iloader cache-status "$MODEL" --name dev-model

iloader start "$MODEL" \
  --name dev-model \
  --local-cache \
  --cache-workers 8 \
  --workers 8
```

### 模拟分布式 rank 读取

`dist-views` 会为每个 rank 生成一个完整 Hugging Face 兼容 view。它不是物理 rank-sharded checkpoint；每个 rank view 仍然暴露完整模型目录。这样做的好处是保持框架默认 loader 不变。

```bash
RANK_VIEWS=$(iloader dist-views "$VIEW" --name smollm2-tp4 --ranks 4 --replace)

iloader dist-simulate "$VIEW" \
  --ranks 4 \
  --rank-view-root "$RANK_VIEWS" \
  --pattern "*.safetensors"
```

分析哪些 tensor 适合未来做框架原生的 rank-sharded checkpoint：

```bash
iloader dist-plan "$MODEL" --tp 4 --limit 20
```

## 命令摘要

```bash
iloader inspect MODEL_DIR
iloader start MODEL_DIR --name NAME
iloader list
iloader status NAME
iloader view NAME
iloader manifest NAME
iloader stop NAME --remove-view
iloader stop --all --remove-view
iloader cache MODEL_DIR --name NAME
iloader cache-status MODEL_DIR --name NAME
iloader cache-list
iloader dist-views MODEL_OR_VIEW --name NAME --ranks N --replace
iloader dist-simulate MODEL_OR_VIEW --ranks N
iloader dist-plan MODEL_DIR --tp N
iloader bench MODEL_OR_VIEW --pattern "*.safetensors" --repeat 3
iloader completion bash
iloader selftest
```

## 实测效果摘要

以下结果来自一台 NVIDIA L20 服务器，使用远端模型存储和本地缓存存储。结果用于说明量级，实际收益取决于模型大小、存储、框架版本、JIT 状态和 OS page cache 状态。

| 场景 | 基线 | InfiniLoader 路径 | 结果 |
| --- | ---: | ---: | --- |
| Bloom-560M safetensors 从 NFS 读 vs 本地 cache | 0.91 GiB/s | 1.98 GiB/s | 读吞吐约 2.18x |
| 多 shard memfd preload，1 worker vs 4 workers | 0.856 s | 0.240 s | preload 约 3.57x |
| vLLM SmolLM2 冷态 NFS 权重阶段 vs memfd view | 33.82 s | 0.15 s | 冷启动大幅改善 |
| vLLM SmolLM2 热态权重阶段 | 0.20 s | 0.15 s | 热态小幅改善 |
| SGLang 0.5.2 SmolLM2 热态 engine load | 12.18 s | 10.37 s | 热重启更快 |
| InfiniLM SmolLM2 load-only 内部日志 | 902.7 ms | 219.5 ms | 权重阶段约 4.1x |
| 4-rank 从 InfiniLoader 派生 rank views 模拟读取 | - | 2.70 GiB / 0.45 s | 聚合约 6.0 GiB/s |

最稳定的收益来自本地 cache 加 memfd view，尤其适合绕过重复 NFS 读取。

## 兼容性说明

- vLLM、SGLang、InfiniLM 都可以通过稳定的 `start/view` 路径使用，通常只需要改模型路径。
- 新版本 vLLM 和 SGLang 具备原生 `sharded_state` loader，是未来支持物理 rank-sharded checkpoint 的优先方向。
- InfiniLM 当前最适合完整 view 路径；rank-sharded checkpoint 需要 InfiniLM adapter 或原生 loader 支持。

## 限制

- 需要 Linux；核心能力依赖 `memfd_create` 和 `/proc/<pid>/fd`。
- 框架读取 view 时，InfiniLoader daemon 必须保持运行。
- 默认 view 会在内存中保留一份选中的模型文件内容。
- 0.1.0 不包含完全透明的同路径拦截；需要把输出的 view 路径传给推理引擎。
- 当前清理版不包含物理 rank-sharded checkpoint 转换；`dist-plan` 只做候选 tensor 切分分析。
