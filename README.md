# InfiniLM

InfiniLM is the high-level inference engine in the InfiniTensor stack. It owns
model execution, runtime management, tensor abstractions, and the Python API.
The lower-level runtime, operator, and collective APIs are provided by
[InfiniRT](https://github.com/InfiniTensor/InfiniRT),
[InfiniOps](https://github.com/InfiniTensor/InfiniOps), and
[InfiniCCL](https://github.com/InfiniTensor/InfiniCCL), respectively.

[InfiniCore](https://github.com/InfiniTensor/InfiniCore) pins those three
projects as submodules. InfiniLM no longer consumes runtime or Python-package
artifacts built by that separate repository; it builds and packages its own
`infinicore` Python module.

## Build

Clone both repositories with their submodules:

```shell
git clone --recurse-submodules https://github.com/InfiniTensor/InfiniCore.git
git clone --recurse-submodules https://github.com/InfiniTensor/InfiniLM.git
```

From InfiniLM, build the NVIDIA dependency stack pinned by the InfiniCore
checkout. The default operator set is the set required by InfiniLM:

```shell
cd InfiniLM
python3 scripts/build_infini_stack.py \
  --infinicore-root ../InfiniCore \
  --cuda-arch sm_80 \
  --jobs 16 \
  --test
export INFINI_ROOT="$PWD/build/integration/nvidia/prefix"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:${LD_LIBRARY_PATH:-}"
```

Then build and install InfiniLM:

```shell
python3 -m pip install . --no-build-isolation
```

Automated migration coverage is limited to NVIDIA, dense non-quantized Qwen3,
and the default static attention implementation. Other platforms and
configurations remain gated for later validation.

The modern operator closure currently supports `qwen3` within that boundary.

## Inference

Run a single-model smoke test:

```shell
python examples/test_infer.py --device nvidia --model=/path/to/model
```

For tensor-parallel inference:

```shell
python examples/test_infer.py --device nvidia --model=/path/to/model --tp=4 --batch-size=16
```

Start the OpenAI-compatible server:

```shell
python python/infinilm/server/inference_server.py --device nvidia --model=/path/to/model --tp=1
```

Paged attention and graph execution are selected by InfiniLM arguments and are
built as part of InfiniLM:

```shell
python examples/bench.py --device nvidia --model=/path/to/model --enable-paged-attn --enable-graph
```

## Development

Format staged files with the repository formatter:

```shell
python scripts/format.py --staged
```

Run the static migration contracts with:

```shell
python -m unittest discover -s test/static -p "test_*.py"
```

## License

InfiniLM is licensed under the MIT License. See [LICENSE](LICENSE).
