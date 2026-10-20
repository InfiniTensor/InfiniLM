# Distributed and hybrid residency validation

Date: 2026-10-18, Asia/Shanghai. Feature branch: `codex/distributed-residency`, based
on reviewed reliability commit `7442271`. No merge into master was performed.

## What was implemented and tested

| User scenario | Implementation and evidence | Remaining boundary |
| --- | --- | --- |
| Each server holds part of the weights | Explicit tensor TP axes and PP stages; streaming safetensors export; four-node TP4, PP4 and TP2×PP2 numerical execution | Minimum **logical tensor payload** for the selected layout, not a proof of global physical RAM optimality. Production engines need semantic adapters. |
| Every server holds all weights for topology changes | Per-node canonical sealed CPU storage; COW tensor views; four-node TP4 → TP2/PP2 → PP4 → TP4 with the original rank filename unavailable | GPU transfers, communications and framework initialization remain; no large-model speedup claim or automatic old-shard redistribution. |
| CPU weights plus GPU hot/CPU cold experts | Grouped, bounded GPU copies; stream-aware leases and LRU; real CPU/GPU ReLU MoE, repeated hot-set changes and failure injection on four DCUs | Sequential reference expert execution; no CPU/GPU overlap claim, distributed EP/MoE, quantized kernel or native KTransformers adapter. |

The detailed protocol, commands, contracts and source references are in
[distributed.md](distributed.md). Sanitized numerical evidence is provided in
[distributed-results.json](distributed-results.json).

## Environments and methodology

- Local: Ubuntu WSL2, Linux 6.18.33.2, Python 3.14.4, PyTorch 2.14.1+cpu,
  safetensors 0.8.0 and NumPy 2.5.4.
- Remote: four distinct authorized SCNet workers, different host and boot
  identities, Linux containers, Python 3.11.9, PyTorch 2.9.0, HIP 6.3.26093.
  Each exposes eight Hygon BW DCUs of approximately 64 GiB; experiments used only
  the one device reserved on each worker. Socket-based RCCL was tested; RDMA was not.
- Existing strict SSH host keys, pod UID and image identity checks were retained.
  Private keys, addresses, hardware serials and lease tokens are not committed.
- Small deterministic four-layer MLP and eight-expert ReLU MoE fixtures were used,
  with independent full CPU reference execution. These are integration and
  correctness tests, not performance results for a production LLM.
- Workers report per-rank model/layout identities, original-path availability,
  tensor and file bytes, PSS/RSS, topology, runtime and timings. Unique host/boot
  identities and complete rank sets were independently checked.

## Results

| Four-node case | Cluster CPU tensor payload | Maximum absolute output error |
| --- | ---: | ---: |
| Gloo TP4 | 68,608 B | 7.45e-9 |
| Gloo PP4 | 67,072 B | 7.45e-9 |
| Gloo TP2×PP2 | 67,584 B | 7.45e-9 |
| Full replicas, four successive TP/PP layouts | 268,288 B | 7.45e-9 in every layout |
| RCCL TP4, one DCU per node | 68,608 B | 7.45e-9 |
| RCCL PP4, one DCU per node | 67,072 B | 7.45e-9 |

Unique MLP payload is 67,072 B. PP4 has no replicated tensor data; TP4 adds 1,536 B
of replicated output biases; TP2×PP2 adds 512 B. The full-replica case is exactly
four times the unique payload. Files, receipts, page cache and runtime memory add
cost beyond this table. Same-node readers share canonical pages; COW writes and
CUDA transfers create additional storage.

CPU TP4/PP4 used code commit `123c6e2`; CPU TP2×PP2 and replica switching, plus
the static RCCL TP4/PP4 runs, used `758f8c8`. GPU TP2×PP2 and GPU dynamic topology
switching were not validated. The exact per-case commits appear in the JSON.

The replica run kept each worker and TensorStore alive for all four layouts and
renamed its original rank weight filename to `.offline` after mapping. Each layout
had exactly one terminal output and passed numerical comparison. This proves the
observed switch reused its RAM mappings; it does not mean every disk copy was
deleted or physically inaccessible.

Four remote devices each passed **11 targeted tests without skips**, including
real GPU expert admission, active-lease protection, capacity rejection, cross-stream
completion and injected transfer failure. Each eight-expert demo retained 512 KiB
on CPU and used a 128 KiB GPU weight pool. Four hot-set phases produced eight GPU
and 24 CPU expert calls, eight promotions and six evictions per node. Maximum
absolute error was **1.79e-7**. PyTorch peak allocated memory was **33,700,352 B**
and peak reserved memory **35,651,584 B**; the 128 KiB weight budget is not a total
application VRAM cap. The demo's CPU source file was unlinked after preload.

Final local regression: **119 passed, 2 real-device tests skipped** in 41.16 seconds.
Both device tests were exercised in the remote 11-test runs. The built
wheel passed Linux `selftest` and `doctor` independently of the source directory.
All six jobs for code commit `758f8c8` passed in
[GitHub Actions run 39107252516](https://github.com/InfiniTensor/InfiniLoader/actions/runs/39107252516):
Python 3.10–3.14 plus the optional PyTorch CPU integration job.

## Failures found and addressed

Real NFS did not support `renameat2(RENAME_NOREPLACE)`. The export path now uses
an atomic exclusive relative symlink to a completed private backing directory on
such filesystems. Independent probes also found publication-after-interrupt and
uncertain `lstat` cases; successful/possibly successful publications are preserved.

Initial multi-node runs had staggered SSH/device startup exceeding the selected
window. A plan/rank/nonce startup barrier now releases workers after preload;
the final slow-environment runs used a 180-second window. The barrier does not
replace a remote liveness service or make arbitrarily delayed startup succeed.

Master-only result collection also missed complete files visible on the owning
nodes. Direct per-node retrieval recovered the original outputs, with matching
startup nonces/PID tokens, and each output was independently recomputed. This was
result reconciliation, not replay of an execution with uncertain completion.
Shared metadata absence must not trigger an automatic duplicate start.

## Independent review and operational closure

Every code commit below was independently reviewed; identified issues were fixed
in unpublished commits before approval and SSH push.

| Commit | Change | Independent review evidence |
| --- | --- | --- |
| `d8fe312` | Strict catalog, layout and streaming export | Two experts; source-race/format probes; 100-test isolated suite |
| `4ab6d4a` | Sealed CPU tensor store and live slicing | Two experts; six tensor tests; daemon-death and final-storage-release probe |
| `674f168` | GPU expert pool and hybrid reference | Two experts; mixed precision/lease/stream checks; later real-device tests |
| `717997e` | Actual TP/PP validation and CPU CI | Validation expert; seven isolated integration tests |
| `123c6e2` | NFS atomic publication and uncertain cleanup | Two experts; 25 tests and post-publication interrupt/EIO probes |
| `758f8c8` | Startup coordination after preload | Two experts; eight isolated integration tests and stale-marker probes |

Two separately identified device reservation rounds were used. All experiment GPU
workers were confirmed exited and every reservation reached closed state with
`cleanup_confirmed=true`. CPU worker cleanup was also checked at completion. No
platform service, host configuration, unrelated process or master branch was changed.
The reservation interface is infrastructure-specific and not part of InfiniLoader.

## Useful next increments

1. Add a concrete production engine adapter, including GQA/packed projections,
   quantization scales and tied-weight semantics; exercise an actual model.
2. Integrate KTransformers/kt-kernel formats and CPU kernels behind its explicit
   ownership contract; profile NUMA placement and conversion/pinning costs.
3. Add asynchronous prefetch and measured overlap, then routing-based hot expert
   selection; compare end-to-end latency against native engine offload.
4. Add authenticated shard transfer, compact per-rank manifests, batched catalog
   preparation and old-topology-to-new-topology redistribution with failure recovery.
5. Extend larger-model, multi-GPU, long-running stress and controlled physical
   memory measurements. Native NVIDIA CUDA, RDMA and distributed expert parallelism
   remain unvalidated by the DCU/socket tests here.
