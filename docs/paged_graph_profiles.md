# Paged graph capture profiles and metadata upload

`EngineConfig` and `InferEngine` accept two independent options:

```python
graph_capture_profiles=[(1, 2048), (8, 4096), (16, 0)]
graph_metadata_upload=False
```

`None` preserves legacy capture policy. An explicit empty list captures no
plans. Explicit lists capture only the unique listed `(batch, KV token bound)`
pairs. Zero means the full physical KV pool. Batches must fit both the cache's
maximum batch and its number of physical pages. Positive bounds must fit the
pool; unsupported bounded model/backend/cache combinations fail before capture.
Current bounded eligibility is ordinary KV prepare, PP1, paged/flash attention,
unquantized KV, 256-token pages and at least two physical pages.

The engine canonicalizes profiles once before creating TP workers. Every worker
receives the same immutable ordered copy. Runtime selection uses the actual
maximum total sequence length and exact batch size. It prefers the smallest
covering positive bound, with zero as a final full-pool choice. Missing batch
and missing bound produce distinct eager fallback reasons in the existing last
execution report. Prefill and all-position sampling retain their eager paths.
No dummy rows or automatic capture are introduced.

Page tables are validated and repacked to the selected graph width. Required
physical page IDs must be in range; unused entries are filled with -1. Invalid
required entries throw before graph storage is mutated. Reset destroys the
compiler and its plans; subsequent compile validates profiles against the new
cache geometry. Profiles themselves remain fixed for the engine lifetime.

The upload option remains opt-in. Each compiler owns one
pinned host slab and one device slab. A replay copies CPU metadata into the slab,
enqueues one H2D and ordered D2D copies to graph inputs, then replays on the same
runtime stream. Original temporary CPU tensors may expire once their bytes are
staged. A serial lease waits for previous queued work before overwriting either
slab; there is no synchronization between upload and graph replay. Compiler
recompile/destruction explicitly drains before graph destinations can disappear.
With profiles left as None and upload disabled, replay retains the original
GPU table fill and narrow synchronous upload. Active CPU pages are validated
without allocating or uploading a full-width host table. Invalid CPU page tables
are rejected before an explicit profile miss can fall back to eager.
Submission exceptions keep the lease live. Failure to establish completion at
destruction terminates rather than releasing an in-use source. General Tensor
copy defaults and existing backend mathematics are unchanged.

For audit logs set `INFINILM_PAGED_GRAPH_STATS=1`. Compiler teardown prints rank,
actual published plan count, staging H2D count and lease drain count. Drains
include explicit teardown, not only steady reuse. Existing last-execution
reports identify `device_graph`, host segments and fallback reasons. These
counters describe the dedicated resource, not all model transfers.

## Local behavior checks

From the repository root:

```text
g++ -std=c++17 -Wall -Wextra -Werror -pedantic test/native/paged_graph_profiles_test.cpp -o paged_graph_profiles_test
g++ -std=c++17 -Wall -Wextra -Werror -pedantic test/native/paged_graph_upload_test.cpp -o paged_graph_upload_test
python test/scripts/paged_graph_config_test.py
```

Run both resulting executables. The staging test uses the production lease with
an asynchronous fake provider that consumes source bytes only on stream drain.
It checks temporary-source lifetime, serial reuse, growth, explicit drain,
normal destruction and exceptions after submission. Policy tests cover B1/B8/B16,
2K/4K/long lengths, boundary and batch changes, empty lists and invalid page IDs.
These checks do not establish CUDA correctness or speed.

## Device validation protocol

Use the existing shared build lock and isolated build/site directories. Compare
legacy policy/upload off, explicit profiles/upload off, legacy policy/upload on,
and explicit profiles/upload on. Keep checkpoint, TP, tokens, output length and
KV capacity/dtype fixed. Test B1/B8/B16 at 2K, 4K and long context, cross-bound
and batch transitions, cache reset and invalid page tables. At each key point
use five adjacent baseline/candidate fresh-process pairs, two warmups and five
measurements per process. Record loaded native SHA256, output equality, NVML v2
peak memory, published plan count, actual graph/host segments, upload/sync counts,
CPU preparation, GPU model time, sampling/readback and full decode wall time.
Do not enable upload by default without stable measured benefit.

The public configuration constructor can also be exercised through the built
native extension on CPU with `test/scripts/paged_graph_native_config_test.py`.
Pass a new task-owned `--scratch` directory and a Qwen3 `--model-config` JSON;
it creates a tiny CPU model configuration and never changes the checkpoint.
`test/native/paged_graph_upload_device_test.cpp` checks the real CUDA upload
lease on two visible devices by default (or one with argument `1`). Run it only
under the shared validation lock after verifying all devices it will use idle.

`test/scripts/paged_graph_device_bench.py` is the task's Qwen3-14B TP2 device
protocol. Pass the existing validation root with `--validation-root` or
`INFINILM_VALIDATION_ROOT`; the checkpoint is read from
`models/Qwen__Qwen3-14B` below that directory. It also requires the intended
isolated site on PYTHONPATH, INFINILM_VALIDATION_SITE, and the two approved UUIDs
in CUDA_VISIBLE_DEVICES. Modes are reference, baseline (candidate with legacy
policy), profiles, upload, combined. Scenario `perf` uses B8 at a total KV length
of 2048, with two warmups and five fixed-shape decode forwards. Other scenarios
cover B1/B8/B16 at total lengths 2048/4096/8192, reset, a batch miss, and candidate
invalid-page rejection. It records CPU argument preparation and synchronous
native decode including sampling/readback; these timings do not separately
attribute GPU compute, sampling, or output readback. That finer attribution
requires a device profiler and must not be inferred from host wall time.


Modes `limited` and `bounded` exercise lists without a full-pool fallback.
Scenario `boundaries` follows each 2K/4K/8K case with the next sequence length,
checking graph selection or the expected KV-bound miss. The negative test uses
an invalid B2 host page table, including the explicit batch-miss path.

`paged_graph_profile_wrapper.py` brackets the third decode with the CUDA
Profiler API. Use Nsight Systems with `--cuda-graph-trace=node` to distinguish
captured model kernels from sampling and readback. A single trace establishes
transfer counts and execution structure, not a latency distribution.
`paged_graph_mixed_wrapper.py` varies tokens and lengths by physical request
row, preserves separate KV pages, and saves the first B16 full logits after an
explicit CUDA fence. Compare all rows and tokens with a separately loaded
reference. Bounded graph widths can change BF16 reduction rounding; report
full-logit errors and argmax equality separately instead of claiming bitwise
identity from equal generated tokens.

Dual A100 validation of Qwen3-14B, TP2, BF16 KV, page size 256 and 544 pages has
completed for the four ablations, boundary/miss cases, reset and invalid page
rejection. Both ranks execute device graphs with zero host segments. The
five-pair performance evidence and actual native hashes are retained in the
task delivery artifacts; this is evidence for that tested configuration only.
Default `None` with upload disabled retains the reference transfer layout.
