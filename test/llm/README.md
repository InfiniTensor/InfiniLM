# Cache and chunking regression tests

Run the CPU regressions with the project Python dependencies installed:

```sh
python -m unittest discover -s test/llm -p 'test_*.py'
```

These tests load isolated Python modules without constructing a native model.
They cover cache ownership/capacity, LRU/SLRU eviction, admission rollback,
remote-KV delayed release, configuration forwarding, chunk boundaries,
phase progress, cancellation and final-only output.

With matching built InfiniLM/InfiniCore extensions and a local dense FP16 model,
run the opt-in A6000 lifecycle check (use `--tp 1` for a single GPU):

```sh
CUDA_VISIBLE_DEVICES=0,1 python test/llm/check_chunk_tp.py \
  --model /path/to/model --tp 2 --chunk-size 300 --policy slru \
  --output /tmp/chunk-tp2.json
```

The check poisons scheduled KV slots, verifies writes on each rank, exercises
shared/repeated prefixes and cancellation, and requires zero final references.
It uses page256/pool16 and compares repeated greedy outputs. `--cache-off`
disables prefix reuse. For PP2 eager, run separate processes with `--tp 1
--pp 2 --stage 0` and `--tp 1 --pp 2 --stage 1`, one visible GPU per process,
matching `--port` values and different output paths.

For actual Decode-graph launch checks, build InfiniCore with `--graph=y` and
preload the counter:

```sh
g++ -std=c++17 -shared -fPIC -I"$INFINI_ROOT/include" \
  test/llm/graph_counter.cc -ldl -o /tmp/infini-graph-counter.so
LD_PRELOAD=/tmp/infini-graph-counter.so CUDA_VISIBLE_DEVICES=0,1 \
  python test/llm/check_chunk_tp.py --model /path/to/model --tp 2 \
  --chunk-size 300 --policy slru --graph --output /tmp/chunk-graph.json
```

`check_chunk_output.py` accepts `--model`, `--tp`, `--chunk-size` and `--output`
to additionally check native output suppression and invalid-input rejection.
KV poisoning is correctness instrumentation, not a performance measurement.

Configuration and limits: [cache and chunking](../../docs/cache-and-chunking.md).
Historical benchmark scripts and measurements are linked from
[PR #573](https://github.com/InfiniTensor/InfiniLM/pull/573).
