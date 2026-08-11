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

Current migration validation is limited to NVIDIA A100 and dense,
non-quantized Qwen3 configurations without linear bias. Qwen3-0.6B has passed
static and paged attention, eager and graph execution, single-request and
batch-2 inference, greedy and non-greedy sampling, TP2, PP2, and combined
TP2+PP2. Paged attention was validated with the default 256-token block size.

Only `qwen3` can be instantiated by the modern model factory. Other model
families, quantized models, and biased Qwen3 configurations remain gated.
Other platforms and custom paged-cache block sizes have not yet been
validated.

When using CoreX PyTorch, export `INFINILM_CXX11_ABI=0` before installing
InfiniLM so that PyTorch, InfiniLM, and the installed Infini stack use the same
libstdc++ ABI:

```shell
export INFINILM_CXX11_ABI=0
```

## Inference

Run a single-model smoke test:

```shell
python examples/test_infer.py --device nvidia --model=/path/to/model
```

For tensor-parallel inference:

```shell
python examples/test_infer.py --device nvidia --model=/path/to/model --tp=2 --batch-size=2
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
