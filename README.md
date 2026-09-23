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

Use Linux with a backend-compatible PyTorch installation, the corresponding GPU
SDK and collective library, CMake, xmake, and GNU binutils (`nm`, `readelf`, and
`c++filt`). Use the same Python environment for building and running InfiniLM.

The default NVIDIA operator configuration also selects external FlashAttention
and FlashInfer shared libraries. Install their providers before building the
stack:

```shell
python3 -m pip install packaging PyYAML ninja
python3 -m pip install "apache-tvm-ffi==0.1.10"
python3 -m pip install "flashinfer-jit-cache==0.6.7" \
  --index-url https://flashinfer.ai/whl/cu130
```

Install a FlashAttention 2 `flash-attn` wheel built for your Python, PyTorch,
CUDA, and libstdc++ ABI. The pinned InfiniOps metadata requires
`flash_attn_2_cuda*.so` and validates its exact exported C++ signatures; it does
not specify a package-version range. A wheel with a matching version number
alone is insufficient if its PyTorch or C++ ABI differs. The FlashInfer sampling
provider accepts `flashinfer-jit-cache>=0.6.7,<0.7` and requires
`apache-tvm-ffi==0.1.10`. These requirements come from the selected InfiniOps
checkout's `src/linked/**/nvidia/*.yaml` files.

The NVIDIA validation image used Python 3.12, PyTorch
`2.10.0a0+b4e4ee81d3.nv25.12`, `flash_attn==2.7.4.post1+25.12`,
`flashinfer-jit-cache==0.6.7+cu130`, and `apache-tvm-ffi==0.1.10`.
The FlashAttention build is supplied by that NVIDIA image. For another
environment, install an ABI-compatible FlashAttention wheel and replace `cu130`
with the CUDA version of its FlashInfer wheel index.

From InfiniLM, build the NVIDIA dependency stack pinned by the InfiniCore
checkout. Before compiling any component, the builder runs InfiniOps' linked
provider resolver for the selected operator slots and checks installed packages,
versions, and library symbols. `--dry-run` prints this preflight command without
executing it:

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

For Iluvatar CoreX, select the Iluvatar component configuration and match the
CoreX PyTorch C++ ABI. Keep the Iluvatar operator configuration outside the
InfiniLM checkout and pass its path explicitly:

```shell
python3 scripts/build_infini_stack.py \
  --infinicore-root ../InfiniCore \
  --backend iluvatar \
  --operator-config /path/to/infiniops_ops_iluvatar.json \
  --jobs 16
export INFINI_ROOT="$PWD/build/integration/iluvatar/prefix"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:${LD_LIBRARY_PATH:-}"
export INFINILM_CXX11_ABI=0
```

For Moore, install MUSA (including muDNN and muBLAS), MCCL, and a compatible
`torch_musa` environment, then select the bundled Moore operator configuration:

```shell
python3 scripts/build_infini_stack.py \
  --infinicore-root ../InfiniCore \
  --backend moore \
  --jobs 16
export INFINI_ROOT="$PWD/build/integration/moore/prefix"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:${LD_LIBRARY_PATH:-}"
```

The Moore configuration supports greedy sampling and does not select the
NVIDIA-only FlashInfer sampling provider. The builder enables `WITH_MCCL` for
Moore. `MUSA_ROOT`, `MUSA_HOME`, or `MUSA_PATH` may select a non-default SDK
installation. Match `INFINILM_CXX11_ABI` to the installed PyTorch build when
needed.

Then build and install InfiniLM:

```shell
python3 -m pip install . --no-build-isolation
```

The InfiniLM wheel includes its two Python extensions, `libinfinicore_runtime`,
and the installed InfiniRT, InfiniOps, and InfiniCCL shared libraries. PyTorch,
FlashAttention, FlashInfer, TVM-FFI, and vendor SDK libraries remain external
runtime dependencies when selected by the operator configuration. Their Python
packages locate native libraries during the build; their presence does not imply
Python operator calls during inference. Keep compatible provider libraries
available to the dynamic loader in the deployment environment. The wheel targets
the Python version, platform, and native ABIs used to build it; it is not a
portable `abi3` or manylinux wheel. On Linux the extension modules resolve Python
symbols from the running interpreter rather than linking directly to `libpython`.

Current migration validation covers NVIDIA A100 and Iluvatar BI-V150 dense,
non-quantized configurations. Real-weight, two-token static-attention smoke
tests have passed for the Baichuan, ChatGLM, FM9G, GLM4, InternLM3, Llama,
MiniCPM4 (`model_type=minicpm` normalized to `minicpm4`), Qwen2, and Qwen3
model families.

Qwen3-0.6B has also passed paged attention, eager and graph execution,
single-request and batch-2 inference, greedy and non-greedy sampling, TP2,
PP2, and combined TP2+PP2. Explicit FlashAttention passed eager and graph
execution on Qwen3-0.6B and BF16 TP4 inference on Qwen3-32B. FlashAttention
also passed eager and graph execution on Llama-3.2-3B. On Iluvatar BI-V150,
9G-8B BF16 real-weight single-GPU inference passed explicit FlashAttention with
InfiniRT graph execution at batch sizes 1, 4, and 16; batch size 64 exceeded the
device's 32 GiB memory capacity during graph compilation. FlashAttention is
enabled on NVIDIA, MetaX, Moore, Cambricon, and Iluvatar. It requires an FP16 or
BF16 model, a head dimension divisible by 8 and no greater than 256, and a paged
KV cache whose block size is a nonzero multiple of 256. Moore and Iluvatar
additionally require a head dimension of 64 or 128.

Qwen3-0.6B with attention, attention-output, and MLP bias enabled passed TP1
and TP2 static and explicit FlashAttention inference, one-time weight
pre-transposition, and TP1 segmented graph replay.

The modern model factory enables `baichuan`, `chatglm`, `fm9g`, `fm9g7b`,
`glm4`, `internlm3`, `llama`, `minicpm`, `minicpm4`, `minicpm_eagle`,
`qwen2`, and `qwen3`. The MiniCPM Eagle path passed NVIDIA A100 MTP inference
with segmented graph replay at batch 16, input length 1024, and output length
256. Iluvatar paged attention passed eager and segmented InfiniRT graph
execution on Qwen3-0.6B; FM9G 8B static and paged TP8 execution paths passed
with weight loading skipped. GPT-2, Mistral, other MoE and multimodal families,
and quantized models remain gated. Backend validation remains model- and
feature-specific.

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

The `--enable-paged-attn` flag selects the paged KV cache layout. With
`--attn=default`, paged caches use the InfiniOps FlashAttention providers and
static caches use `static-attn`. The removed `paged-attn` backend is no longer
a valid `--attn` value. Enable paged cache and graph execution with:

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
