# InfiniLoader

[中文](README.zh-CN.md)

The development branch adds model identity checks, single-owner startup, sealed
memfd files, versioned disk caches, container-aware budgets, and lease-aware cache
management. See [operations and migration](docs/operations.md) for `doctor`,
`verify`, `run`, `gc`, and container guidance.

```bash
iloader doctor
iloader start "$MODEL" --name dev --local-cache \
  --max-memory-bytes 32GiB --reserve-bytes 4GiB --max-resident-bytes 64GiB
iloader run dev -- vllm serve '{model}'
iloader gc --resident --max-bytes 32GiB       # preview
iloader gc --resident --max-bytes 32GiB --apply
```

Use `iloader cache` plus `iloader run --cache-path PATH -- COMMAND` for disk-only
caching. Existing direct-path readers remain supported; eviction protection requires
`run`, so leave automatic idle eviction disabled for unregistered readers.

`InfiniLoader` is a userspace weight loading manager for LLM inference
framework development. It is designed for the loop where you repeatedly restart
vLLM, SGLang, InfiniLM, or a custom engine while debugging loader, kernel,
scheduler, or runtime code, and every restart pays the cost of reading model
weights from storage again.

The default mode keeps a model directory resident in memory and exposes a normal
filesystem-looking Hugging Face model directory. Frameworks keep using their
existing file loaders; in the common path you only change the model path.

## Quick Start

The basic path is a memfd-backed, memfs-style in-memory view. If you usually run
a framework against `/path/to/huggingface-model`, add one command before the
framework launch and use the returned path instead.

```bash
cd InfiniLoader
source env.sh

MODEL=/path/to/huggingface-model
MODEL_VIEW=$(iloader start "$MODEL" --name dev-model)
```

Then pass `"$MODEL_VIEW"` to the inference engine as the model path. The
InfiniLoader command is the same; only the engine launch command changes.

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

When you are done:

```bash
iloader stop dev-model --remove-view
```

That is the stable compatibility mode for 0.1.0: no framework loader code
changes, only the model path changes. vLLM, SGLang, InfiniLM, local cache,
distributed views, progress/status, and completion are covered below.

## What It Solves

Repeated LLM engine debugging often spends more time loading weights than
running the code under test:

- network filesystems can be slow or bursty for large safetensors shards;
- hot OS page cache helps, but does not remove framework parsing and H2D costs;
- distributed runs may cause every rank to scan or read the same checkpoint;
- modifying framework code should not require editing model loading APIs.

`InfiniLoader` provides a long-lived loader service that can outlive framework
processes. You can stop and restart vLLM/SGLang/InfiniLM while the weight view
stays warm.

## How It Works

The default `start/view` path is Linux-only and does not require root or FUSE:

1. A background daemon reads selected files from a model directory.
2. Each file is copied into an anonymous `memfd_create` file.
3. The daemon keeps those file descriptors open.
4. A view directory mirrors the original model tree with symlinks to
   `/proc/<daemon-pid>/fd/<fd>`.
5. Frameworks use ordinary `open`, `stat`, and `mmap` calls against that view.

Optional stable layers are available for local disk caching, distributed rank
views, and tensor-parallel split analysis.

## Setup And Runtime State

`env.sh` adds the repository `src` directory to `PYTHONPATH` and defines an
`iloader` shell function for the current terminal. It does not require a wrapper
script under `bin`. In bash, it also registers command completion for iloader
subcommands, options, and active cache names.

You can also install InfiniLoader as a Python package:

```bash
pip install -e .
iloader selftest
```

Run the unit test suite with:

```bash
pip install -e .[test]
python -m pytest
```

Runtime state is placed under `$ILOADER_BASE_DIR` when it is set, otherwise
under `$XDG_CACHE_HOME/infiniloader` or `~/.cache/infiniloader`. Every command
that creates runtime state also accepts `--base-dir`.

## Command Surface

The public 0.1.0 interface is intentionally small:

- Primary workflow: `start`, `list`, `status`, `view`, `stop`.
- Utilities: `inspect`, `bench`, `completion`, `selftest`.
- Advanced but stable helpers: `cache`, `cache-status`, `cache-list`,
  `dist-views`, `dist-simulate`, `dist-plan`.

`serve` is an internal daemon entry used by `start` and is hidden from help and
completion. Transparent same-path interception, global mount management, and
GPU-resident tensor caches are not part of the 0.1.0 CLI; the supported contract
is to pass the view path printed by `iloader start` to the inference engine.

## Detailed Usage

### Basic Memfd Cache

The default `start` command preloads selected model files into anonymous memory
files and prints a Hugging Face-compatible view path to stdout. Progress,
throughput, and ETA are printed to stderr while the daemon is loading.

```bash
MODEL_VIEW=$(iloader start "$MODEL" --name dev-model)
```

Use the view path with any framework command that accepts a model directory:

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

From another terminal, you can inspect and manage active resident caches:

```bash
iloader list
iloader status --name dev-model
iloader stop dev-model --remove-view
```

Stopping a cache that is still `starting` or `loading` cancels the preload.

### Keep a Model Warm While Restarting a Framework

```bash
MODEL_VIEW=$(iloader start "$MODEL" --name dev-model)

# Repeat this while editing framework code.
vllm serve "$MODEL_VIEW" --tensor-parallel-size 1
```

Stopping `vllm serve` does not stop the InfiniLoader daemon. The next framework
restart reuses the same warm view. If the daemon is already running,
`iloader start "$MODEL" --name dev-model` simply prints the existing view path.

### Materialize a Local Cache First

This is useful when the original model directory lives on remote or shared
storage and you want InfiniLoader to build a reusable local cache before
creating the in-memory view.

```bash
iloader cache "$MODEL" --name dev-model --cache-workers 8
iloader cache-status "$MODEL" --name dev-model

iloader start "$MODEL" \
  --name dev-model \
  --local-cache \
  --cache-workers 8 \
  --workers 8
```

### Simulate Distributed Rank Reads

`dist-views` creates one full Hugging Face-compatible view per rank. These are
not physically sharded checkpoints; every rank view still exposes the complete
model tree. This keeps framework loaders unchanged and is useful for launchers
that expect per-rank paths.

```bash
RANK_VIEWS=$(iloader dist-views "$VIEW" --name smollm2-tp4 --ranks 4 --replace)

iloader dist-simulate "$VIEW" \
  --ranks 4 \
  --rank-view-root "$RANK_VIEWS" \
  --pattern "*.safetensors"
```

To inspect which tensors could be split for a future engine-native
rank-sharded checkpoint:

```bash
iloader dist-plan "$MODEL" --tp 4 --limit 20
```

## Command Summary

Explicit tensor sharding, live CPU layout selection and the CPU/GPU expert API
are documented in [Distributed and hybrid residency](docs/distributed.md).

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

## Measured Results

Measurements below were collected on a single NVIDIA L20 server using remote
model storage plus local cache storage. They are meant as order-of-magnitude
guidance; exact results depend on model size, storage, framework version, JIT
state, and OS page cache state.

| Scenario | Baseline | InfiniLoader path | Result |
| --- | ---: | ---: | --- |
| Bloom-560M safetensors read from NFS vs local cache | 0.91 GiB/s | 1.98 GiB/s | 2.18x read throughput |
| Multi-shard memfd preload with 1 vs 4 workers | 0.856 s | 0.240 s | 3.57x preload speedup |
| vLLM SmolLM2 cold-ish NFS weight stage vs memfd view | 33.82 s | 0.15 s | large cold-start improvement |
| vLLM SmolLM2 hot weight stage | 0.20 s | 0.15 s | small hot-cache improvement |
| SGLang 0.5.2 SmolLM2 hot engine load | 12.18 s | 10.37 s | faster hot reload |
| InfiniLM SmolLM2 internal load-only log | 902.7 ms | 219.5 ms | 4.1x weight-stage improvement |
| 4-rank simulated read from InfiniLoader-derived rank views | - | 2.70 GiB in 0.45 s | about 6.0 GiB/s aggregate |

The most stable production-like gain comes from avoiding repeated NFS reads via
local cache plus memfd views.

## Compatibility Notes

- vLLM, SGLang, and InfiniLM work with the stable `start/view` path by changing
  the model path.
- vLLM and SGLang have native `sharded_state` loaders in recent versions, which
  are the preferred path for future physical rank-sharded checkpoints.
- InfiniLM currently fits the full-view path best. Rank-sharded checkpoint
  loading would need an InfiniLM-specific adapter or native loader support.

## Limitations

- Linux is required for `memfd_create` and `/proc/<pid>/fd` views.
- The daemon must stay alive while framework processes read the view.
- The default view duplicates selected model bytes in RAM.
- Fully transparent same-path interception is not included in 0.1.0; pass the
  printed view path to the engine.
- `shard plan/export/verify` supports explicit InfiniLoader TP/PP checkpoints.
  `dist-plan` still only analyzes candidates. Loading these checkpoints into a
  production model engine requires an adapter; see the reference PyTorch MLP.
