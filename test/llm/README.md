# Cache and chunking regression tests

Run the CPU regressions with the project Python dependencies installed:

```sh
python -m unittest discover -s test/llm -p 'test_*.py'
```

These tests load isolated Python modules without constructing a native model.
They cover cache ownership/capacity, LRU/SLRU eviction, admission rollback,
remote-KV delayed release, configuration forwarding, chunk boundaries,
phase progress, cancellation and final-only output.

With matching built InfiniLM/InfiniCore extensions and a dense FP16 model,
run the short native regression (use `--tp 1` for a single GPU):

```sh
CUDA_VISIBLE_DEVICES=0,1 python test/llm/check_chunk_output.py \
  --model /path/to/model --tp 2 --chunk-size 17
```

The check compares chunked and ordinary greedy output, verifies native
intermediate-output suppression and invalid-input rejection, and exercises
prefix reuse, cancellation and complete page-reference reclamation. Add `--graph`
to run ordinary Decode with graphs; compile InfiniCore with `--graph=y` first.
The model must accept token IDs 1–67 and meet the scheduler’s minimum
`max_position_embeddings` of 1024.

Longer TP/PP experiments, KV poisoning and graph-launch interception are archived
in the [validation tools](https://github.com/big-hip/InfiniCore/tree/b635f35f359d2f536b9ba5ca82686b6b2b988cb7/docs/validation/cache-chunk-tools-20260921).

Configuration and limits: [cache and chunking](../../docs/cache-and-chunking.md).
Historical benchmark scripts and measurements are linked from
[PR #573](https://github.com/InfiniTensor/InfiniLM/pull/573).
