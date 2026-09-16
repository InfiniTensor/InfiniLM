# Prefix-cache benchmark

`benchmark_prefix_cache.py` is the common baseline/candidate harness for the
paged prefix-cache LRU experiment. It has two deliberately separate modes.
`metadata` loads only the Python cache metadata modules and never imports the
native InfiniLM package. `model` uses the normal `AsyncLLMEngine` API and actual
token IDs on CUDA.

## CPU metadata mode

Run the same copied harness in each checkout. Pool setup, hash preparation, and
JSON serialization are outside the timed calls. The JSON reports the median,
p95, and valid sample count for usable-capacity queries and reclaiming 1, 8, or
64 blocks at 0%, 50%, and 90% pinned occupancy. It separately records the
actual reclaimed count when the requested reclaim is impossible. `tracemalloc`
reports Python peak/current increments after constructing and filling the pool;
these values do not include GPU memory.

```bash
experiment_dir=/path/to/pr1-lru-results
for blocks in 512 4096 65536; do
  conda run -n infinilm python test/llm/benchmark_prefix_cache.py \
    --mode metadata --num-blocks "$blocks" --repeat 5 --seed 0 \
    --output "${experiment_dir}/candidate-metadata-${blocks}.json"
done
```

Use `baseline-metadata-${blocks}.json` for the detached baseline checkout. The
policy examples are CPU control-plane observations. They demonstrate victim and
subsequent hash-hit differences; they are not GPU speed measurements.
Metadata setup assigns distinct synthetic 16-byte block hashes rather than
timing real `xxhash` calculation. This isolates lookup/reclaim metadata cost;
the synthetic hashes are prepared before every timed call.

## CUDA model mode

Model mode fixes TP=1, FP16, paged attention, eager execution, block size 256,
32 generated tokens, and maximum batch size 4. It accepts only the documented
scenarios and concurrency values. The first 16 requests warm the cache and are
retained in the trace but excluded from the performance denominator. Each
repeat runs in a fresh child process so native model state cannot leak between
repeats. A timeout, child failure, missing token stream, or output other than 32
tokens makes the experiment fail and writes a failure JSON.

```bash
model_dir=/path/to/Qwen2.5-1.5B-pr1-fp16
experiment_dir=/path/to/pr1-lru-results
export INFINILM_MODEL_PROVENANCE="${experiment_dir}/model-experiments.json"
export INFINILM_BUILD_PROVENANCE="${experiment_dir}/build-provenance.json"
python test/llm/benchmark_prefix_cache.py \
  --mode model \
  --model "$model_dir" \
  --prefix-cache on --scenario hot-cold --concurrency 1 \
  --num-blocks 64 --repeat 3 --repeat-timeout-seconds 3600 --seed 0 \
  --output "${experiment_dir}/candidate-1.5b-hot-cold-cache-on-blocks-64.json"
```

Set `INFINILM_MODEL_PROVENANCE` to the controller's JSON manifest containing
the model ID, immutable revision, and FP16 overlay hashes. The harness embeds
the matching manifest entry without assuming a machine-specific research path.
Set `INFINILM_BUILD_PROVENANCE` to the native build manifest. The harness
embeds and hashes it, hashes both imported extension binaries, and requires the
manifest artifact hashes to match those binaries.

Repeat with cache `on` and `off`, pools 64 and 512, and scenarios `hot-cold`,
`hot-shift`, `no-reuse`, `over-capacity`, `mixed-length`, and `shared`.
`shared` should use concurrency 4; other scenarios support 1 or 4. The
controller is responsible for the complete model matrix and isolated native
library environment.

The result preserves the complete prompt-token trace and SHA256, every output
token ID and delivery timestamp-derived metric, admitted local-cache and
prefill token counts, grouped TTFT, aggregate throughput, both repository SHAs,
actual imported source paths, GPU/driver/CUDA details, and FP16 overlay
provenance. Throughput is total delivered measurement tokens divided by the
shared measurement wall time. Delivery intervals include engine threads and
queues and must not be described as GPU kernel time.

For an exactly 256-aligned prompt, the scheduler intentionally retains the last
token rule at block granularity: a 1024-token prompt can reuse only 768 tokens.
The benchmark workloads add a 32-token unique tail to a 1024-token prefix, so an
admitted full-prefix hit reports 1024 cached tokens and 32 prefill tokens.

## Result artifacts

Final quiet-window measurements are a Task 4 deliverable. Populate the table
with the controller's accepted artifacts; do not copy functional-smoke timings
into performance claims.

| Measurement | Baseline artifact | Candidate artifact | Result |
|---|---|---|---|
| Metadata, 512/4096/65536 blocks | pending Task 4 | pending Task 4 | pending |
| Qwen2.5-1.5B correctness smoke | pending Task 4 | pending Task 4 | pending |
| Qwen2.5-7B scenario matrix | pending Task 4 | pending Task 4 | pending |
