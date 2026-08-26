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
non-quantized configurations. Real-weight, two-token static-attention smoke
tests have passed for the Baichuan, ChatGLM, FM9G, GLM4, InternLM3, Llama,
MiniCPM4 (`model_type=minicpm` normalized to `minicpm4`), Qwen2, and Qwen3
model families.

Qwen3-0.6B has also passed paged attention, eager and graph execution,
single-request and batch-2 inference, greedy and non-greedy sampling, TP2,
PP2, and combined TP2+PP2. Explicit FlashAttention passed eager and graph
execution on Qwen3-0.6B and BF16 TP4 inference on Qwen3-32B. FlashAttention
also passed eager and graph execution on Llama-3.2-3B. It requires NVIDIA, an
FP16 or BF16 model, a head dimension divisible by 8 and no greater than 256,
and a paged KV cache whose block size is a nonzero multiple of 256.

Qwen3-0.6B with attention, attention-output, and MLP bias enabled passed TP1
and TP2 static and explicit FlashAttention inference, one-time weight
pre-transposition, and TP1 segmented graph replay.

The modern model factory enables `baichuan`, `chatglm`, `fm9g`, `fm9g7b`,
`glm4`, `internlm3`, `llama`, `minicpm`, `minicpm4`, `qwen2`, and
`qwen3`. GPT-2, Mistral, MoE and multimodal families, and quantized models
remain gated. Other platforms have not yet been validated.

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

Select the linked InfiniOps FlashAttention providers explicitly with a paged
KV cache:

```shell
python examples/test_infer.py --device nvidia --model=/path/to/model --enable-paged-attn --attn=flash-attn
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
