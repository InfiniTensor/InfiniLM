# Distributed CPU residency and hybrid expert integration

InfiniLoader now provides explicit safetensors partitioning, a PyTorch CPU tensor
store, and a bounded GPU expert pool. The reference examples execute real TP/PP
and a dense ReLU MoE. Framework-specific Llama/GQA, quantized kernels and
KTransformers integration remain separate adapter work.

## When to use each mode

All three modes address real workloads:

| Mode | Suitable workload | Capacity and tradeoff |
| --- | --- | --- |
| Sharded CPU weights | Fixed TP/PP deployment; models larger than one node's RAM | Cluster payload `U + R`: unique tensor bytes plus execution-required replicas. PP can achieve `R=0`; TP often replicates some parameters. A new topology may require preparing/transferring new shards. |
| Full CPU replica per node | Repeated debugging, worker restart, frequent TP/PP changes | `N × U` for N nodes, with readers on a node sharing its canonical store. Slice existing CPU mappings for new layouts; GPU copies, collectives and engine initialization still cost time. |
| CPU canonical store plus GPU hot experts | Sparse MoE inference with limited VRAM | CPU retains the full model, GPU retains selected complete expert groups. Cold experts can compute on CPU. Eviction drops the GPU copy; immutable weights need no D2H writeback. |

`U` counts actual validated checkpoint tensor payload, including separately stored
quantization tensors and duplicate stored parameters. It does not deduplicate
semantic tied weights. `R` is dictated by the explicit execution layout. These are
logical capacities, not guarantees of minimum physical system RAM: file headers,
page cache, Python, COW pages, activations, staging and allocator state are additional.
Use PSS plus cgroup/system measurements rather than summing shared-page RSS.

The demand is independently supported by [vLLM's rank-specific sharded loader](https://docs.vllm.ai/en/latest/api/vllm/model_executor/model_loader/sharded_state_loader/),
[PyTorch distributed checkpoint resharding](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html),
and [KTransformers' CPU/GPU heterogeneous expert placement](https://github.com/kvcache-ai/ktransformers).
The policy choices above are our interpretation of those workloads, not a claim
that these engines consume the new InfiniLoader shard format directly.

## Explicit checkpoint preparation

The legacy `dist-plan` remains a heuristic report. The new executable path requires
an exact tensor inventory and explicit TP dimension/PP stage for **every** tensor:

```json
{
  "format": 1, "tp": 2, "pp": 1,
  "tensors": {
    "up.weight": {"stage": 0, "split_dim": 0},
    "down.weight": {"stage": 0, "split_dim": 1},
    "down.bias": {"stage": 0, "split_dim": null}
  }
}
```

```bash
iloader shard plan /models/source --layout layout.json --mode sharded --output plan.json
iloader shard export /models/source --plan plan.json --rank 0 --output /cache/rank0
iloader shard verify /cache/rank0
```

`--mode replicated` exports a complete canonical checkpoint for each node. A plain
full safetensors model can also be preloaded directly without exporting replicas.
Use one daemon per node. Each reader opens its own TensorStore over that daemon's
canonical memfds, sharing the underlying CPU pages across processes. A daemon per
local rank unnecessarily duplicates the canonical CPU allocation.

Plans carry path-independent SHA-256 content identity and a separate layout ID.
Catalogs reject duplicate tensors/JSON keys, invalid offsets, shapes, dtypes and
index mismatches. Nonzero-axis slicing streams the appropriate row intervals.
Preparation performs multiple full source hashes to reject concurrent changes;
this is an offline cost, **not** a claim that preparation reads only rank bytes.
Export uses bounded chunks; the destination contains only that rank's tensors.

On filesystems without Linux `RENAME_NOREPLACE`, publication atomically creates a
relative rank symlink to a completed private `.iloader-export-*` backing directory.
Keep the link and its backing together. To copy a rank independently, dereference
the link (for example `cp -aL rank0 copy-of-rank0`). If publication is interrupted,
verify an existing destination before retrying; existing output is never replaced.
An uncertain NFS lookup preserves possibly published data for reconciliation.

The output is an **InfiniLoader explicit-layout checkpoint**. Being valid
safetensors does not make it a native HF/vLLM/InfiniLM checkpoint. A model adapter
must implement the specified collectives, packed projections, GQA, biases and
quantization semantics. No name heuristic silently decides these semantics.

## Use the canonical CPU store

Install a PyTorch build appropriate for the hardware, then install this project.
Core commands still have no PyTorch dependency.

```bash
iloader start /cache/rank0 --name rank0 --workers 1 --chunk-size 1048576
iloader run rank0 -- python adapter.py '{model}'
```

```python
from pathlib import Path
import sys
from infiniloader.shards import read_json
from infiniloader.torch_store import TensorStore

resident_view = sys.argv[1]
layout = read_json(Path("plan.json"))["layout"]
with TensorStore(resident_view) as store:
    tensors = store.rank_tensors(layout, rank=0)
    # Pass CPU tensors to an adapter; .to(device) transfers only required slices.
```

The store accepts sealed InfiniLoader memfds, validates their contents and maps
them privately with COW. Reads share canonical pages; in-place writes consume
private pages without modifying the daemon. `torch.frombuffer` retains the mapping
through tensor storage lifetime. Closing a store prevents new reads but does not
invalidate already returned tensors/views. Completed mappings also survive daemon
exit; new opens through a dead view do not. Such client-held mappings can outlive
the daemon's logical reservation and continue using physical RAM.

Use `iloader run` to protect initialization from daemon idle expiry/stop. The
TensorStore path API itself does not automatically acquire a daemon lease.

A replicated store can call `rank_tensors` repeatedly with different valid
layouts without rereading the original source or rebuilding CPU weights. A
pre-sharded store is bound to its original layout and rank. Redistribution from
an already sharded old topology to another sharded topology is not implemented.

## CPU/GPU expert integration

```python
from infiniloader.experts import ExpertPool

groups = {"0": {"up": "experts.0.up", "down": "experts.0.down"}}
with ExpertPool(store, groups, max_gpu_bytes=256 * 1024**2, device="cuda:0") as pool:
    pool.promote("0")  # Whole-group synchronous upload; False means CPU fallback.
    with pool.acquire("0") as lease:
        output = engine_expert_kernel(inputs.to(lease.device), lease.weights)
```

This API is for **inference**. Do not retain weights or submit work after a lease
ends. Autograd that saves weights beyond the lease is outside the contract. GPU
weights default to the current consumer stream; use `acquire(..., stream=stream)`
and `lease.record_stream(other_stream)` for every additional consumer stream.
Before eviction, the pool waits for recorded work and protects allocator reuse.
Active leases cannot be evicted. Failed uploads never publish half an expert.

The budget counts pool-owned weight payload, including admission in progress; it
does not cap activations, external tensor references, CUDA context or allocator
reserved memory. Record both weight bytes and PyTorch allocated/reserved peaks.
Promotions are currently synchronous; asynchronous prefetch and routing prediction
are future work. The caller selects hot experts, while the pool supplies admission,
CPU fallback and lease-safe LRU replacement.

`hybrid_moe` is a reference dense ReLU execution path, with top-k routing and
mixed CPU/GPU output aggregation. Native KTransformers/kt-kernel uses specialized
formats and kernels; its conversion/NUMA/kernel adapter has not been implemented.
Pinning, quantization conversion and CPU→GPU transfer are not zero-copy operations.

## Reproduce numerical validation

Install `pytest`, `safetensors` and `numpy` beside the correct PyTorch build.

```bash
python -m pytest -q
python examples/distributed_mlp.py local --root /tmp/tp4-new --mode sharded --tp 4 --pp 1 --output tp4.json
python examples/distributed_mlp.py local --root /tmp/switch-new --mode replicated --tp 4 --pp 1 --output switch.json
python examples/hybrid_moe.py --device cuda:0 --output hybrid.json
```

The MLP fixture uses true all-reduce and point-to-point activation transfers,
including correct post-reduction bias addition. Replica switching removes the
source filename after preload and requires every topology to produce exactly one
numerically checked terminal output. The fixture is intentionally small and does
not establish large-model speedup or optimal physical RAM.

For multiple hosts, prepare on shared storage or transfer each rank checkpoint,
then invoke one `worker` per host with the same rendezvous and different rank:

```bash
python examples/distributed_mlp.py prepare --root /shared/new-run --mode sharded --tp 2 --pp 2
mkdir -m 700 /shared/new-run/startup
# Run separately on hosts 0..3, substituting RANK and a reachable rendezvous address.
python examples/distributed_mlp.py worker --root /shared/new-run --rank RANK --world-size 4 \
  --init tcp://HOST0:PORT --backend gloo --device cpu --output /shared/new-run/resultRANK.json \
  --ready-dir /shared/new-run/startup --timeout 180
# After launching all workers, release them from one trusted controller:
PYTHONPATH=src:examples python - <<'PY'
from pathlib import Path
from distributed_mlp import publish_start
from infiniloader.shards import read_json
root = Path('/shared/new-run')
plan = read_json(root / 'fixture.json')['plan']
publish_start(root / 'startup', 4, plan['plan_id'], 180)
PY
```

Use a fresh private barrier directory and a single controller. The participant
nonces prevent joining an old release; the recorded PID/token is diagnostic and
does not authenticate remote liveness. The startup and collective waits are both
bounded by `--timeout`; choose a window that covers slow SSH and device warmup.
Collect results from each owning node. In the tested NFS environment, the master
temporarily missed records that were complete on the generating node; absence
from a shared-directory lookup did not establish that execution had failed.

For GPU communication use the device/framework's `nccl` backend (RCCL under ROCm)
and an assigned device such as `cuda:0`. Memfd views must remain node-local; shared
NFS paths do not make another host's `/proc/PID/fd` valid. External launchers must
verify host/checkpoint identity, allocate devices, bound execution and collect all
rank results. The example is a validation harness, not a fault-tolerant cluster
controller, authenticated tensor-transfer service or inference serving framework.

See [distributed validation](distributed-validation.md) for the tested hardware,
results, independent review and remaining limitations.
