# InfiniLM MetaX C500 basic workflow

This runbook reproduces basic Qwen3-0.6B inference on a fresh MetaX C500
container. It tracks each repository's remote default branch; older C550 commit
pins are not used.

## Validation record

Date: 2026-11-24.

Environment:

- GPU: 1 x MetaX C500, MACA `3.5.3.20`, driver `3.8.30`
- Architecture: `/opt/maca/bin/__metaxgpu_arch` reports `xcore1000`
- Python 3.10.10, PyTorch `2.8.0+metax3.5.3.9`, C++11 ABI `1`
- Tools installed by this run: CMake `3.22.1`, xmake `3.1.2+20261124`

Revisions recorded during validation, not used as pins:

- InfiniLM `12cc072a5c061349d785da88df9c34c290a73664`
- InfiniCore `08d972c62e35f6a3c117191927aca853bc151c4b`
- InfiniRT `ce63347d2b96393109fe15302589c938ef70782e`
- InfiniOps `c5ae2f474f5d1848af8371dafd56f2e87977fd1f`
- InfiniCCL `eed517b0a60d820984cdec86773742364665c4d9`

Result:

- InfiniRT CPU+MetaX: build and install PASS
- InfiniOps C500 basic profile: build and install PASS
- InfiniCCL CPU+MetaX/MCCL: build and install PASS
- InfiniLM runtime and `_infinilm`: build and install PASS
- Qwen3-0.6B static-cache greedy inference: PASS
- Clean 16-token output began `<think>\nOkay, the user just said "Hello".`
- Generation time: `1007.7 ms`; GPU memory released after exit

## Scope

This basic path uses `--attn=default`, static KV cache, and greedy decoding
(`top_k=1`). Input tensors are created on CPU and model operators run on MetaX.

FlashAttention, paged KV cache, non-greedy sampling, and multi-card TP/PP are
outside this run. The profile keeps flash and top-k/top-p APIs linkable but has
no executable provider for them.

On MACA `3.5.3` with PyTorch `2.8`, one `mxcc` translation unit cannot include
both native MetaX device headers and Torch linked-provider headers: errors include
`Unexpected branch` in `c10/macros/Macros.h` and a missing `__hneg` overload. The
basic profile therefore keeps native slot 0 implementations and marks linked-only
providers `public_api_only`. No source-code patch is needed for this path.

## Prepare the machine

Use the operator-provided SSH credential; do not write the password into the
repository or shell history. Keep all work under `/data`, which is persistent.

```bash
mx-smi
/opt/maca/bin/__metaxgpu_arch
python - <<'PY'
import torch
print(torch.__version__)
print(torch._C._GLIBCXX_USE_CXX11_ABI)
PY
df -h /data
```

Install the missing host tools:

```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  cmake ca-certificates curl git build-essential
curl -fsSL https://xmake.io/shget.text -o /tmp/xmake-install.sh
bash /tmp/xmake-install.sh
rm -f /tmp/xmake-install.sh
source ~/.xmake/profile
```

Set `XMAKE_ROOT=y` because this container runs as root.

## Clone latest default branches

```bash
git clone git@github.com:InfiniTensor/InfiniLM.git /data/InfiniLM
cd /data/InfiniLM
git submodule update --init --recursive third_party
git switch main
git pull --ff-only origin main
git -C third_party/InfiniCore switch main
git -C third_party/InfiniCore pull --ff-only origin main
git -C third_party/InfiniCore/submodules/InfiniRT switch master
git -C third_party/InfiniCore/submodules/InfiniRT pull --ff-only origin master
git -C third_party/InfiniCore/submodules/InfiniOps switch master
git -C third_party/InfiniCore/submodules/InfiniOps pull --ff-only origin master
git -C third_party/InfiniCore/submodules/InfiniCCL switch master
git -C third_party/InfiniCore/submodules/InfiniCCL pull --ff-only origin master
```

Record the current revisions with `git rev-parse HEAD` in each repository. The
top-level checkout can show `third_party/InfiniCore` modified because its gitlink
was intentionally moved to live InfiniCore `main`; preserve that state.

## Common environment

Use this in every fresh build shell:

```bash
cd /data/InfiniLM
export WORK=/data/InfiniLM
export CORE=$WORK/third_party/InfiniCore
export PREFIX=$WORK/prefix
export OPS_JSON=$WORK/docs/infiniops_ops_c500_basic.json
export MACA_PATH=/opt/maca
export XMAKE_ROOT=y
export INFINI_ROOT=$PREFIX
export INFINILM_CXX11_ABI=1
export PATH=/opt/conda/bin:/root/.local/bin:/opt/maca/bin:/opt/maca/mxgpu_llvm/bin:/opt/maca/ompi/bin:/opt/maca/ucx/bin:/opt/mxdriver/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CPATH=/opt/conda/include/python3.10:/opt/maca/include
export CPLUS_INCLUDE_PATH=/opt/conda/include/python3.10:/opt/maca/include
export LIBRARY_PATH=/opt/maca/lib
export PYBIND11_INCLUDE=/opt/conda/lib/python3.10/site-packages/pybind11/include
export PYTHON_INCLUDE_DIR=/opt/conda/include/python3.10
```

The operator profile is `docs/infiniops_ops_c500_basic.json`. It is derived from
the upstream dense no-Marlin profile: executable operators use slot 0, while
flash, linked-only, unused backend-specific, and non-greedy sampling APIs are
public-only. Validate it with:

```bash
python -m json.tool docs/infiniops_ops_c500_basic.json >/dev/null
```

## Build InfiniRT

CPU must remain enabled. `from_list()` creates a CPU runtime even when model
execution is MetaX; MetaX-only InfiniRT makes CPU count zero and causes `device
index is unavailable for the requested backend`.

```bash
cmake -S "$CORE/submodules/InfiniRT" -B "$WORK/build/rt" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_CXX_FLAGS='-D_GLIBCXX_USE_CXX11_ABI=1' \
  -DWITH_CPU=ON -DWITH_METAX=ON
cmake --build "$WORK/build/rt" -j16
cmake --build "$WORK/build/rt" --target install -j16
```

Current InfiniRT registers no CTest cases. Gate on successful build/install;
generated switch warnings are expected.

## Build InfiniOps

```bash
cmake -S "$CORE/submodules/InfiniOps" -B "$WORK/build/ops" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_C_COMPILER="$CORE/submodules/InfiniOps/scripts/mxcc_wrapper.sh" \
  -DCMAKE_CXX_COMPILER="$CORE/submodules/InfiniOps/scripts/mxcc_wrapper.sh" \
  -DCMAKE_CXX_FLAGS='-D_GLIBCXX_USE_CXX11_ABI=1' \
  -DWITH_CPU=ON -DWITH_METAX=ON -DWITH_TORCH=ON -DWITH_LINKED=ON \
  -DINFINI_OPS_OPS="$OPS_JSON" \
  -DINFINI_RT_ROOT="$PREFIX" \
  -DINFINI_RT_INCLUDE_DIRS="$PREFIX/include" \
  -DINFINI_RT_LIBRARY="$PREFIX/lib/libinfinirt.so" \
  -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON="$WORK/third_party/json"
cmake --build "$WORK/build/ops" --target infiniops -j16
cmake --build "$WORK/build/ops" --target install -j16
```

Expected configure evidence:

```text
PyTorch CXX11 ABI: 1
Operator selection: /data/InfiniLM/docs/infiniops_ops_c500_basic.json
Resolved 0 linked operator source(s).
Generating wrappers - done
```

A missing `clang-format` warning is harmless. The install target may build the
four current examples; this succeeded on C500.

## Build InfiniCCL

```bash
cmake -S "$CORE/submodules/InfiniCCL" -B "$WORK/build/ccl" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_CXX_FLAGS='-D_GLIBCXX_USE_CXX11_ABI=1' \
  -DWITH_METAX=ON -DWITH_MCCL=ON \
  -DWITH_NCCL=OFF -DWITH_OMPI=OFF -DWITH_MPICH=OFF
cmake --build "$WORK/build/ccl" --target install -j16
```

Expected configure evidence: `Devices [cpu, metax] | Backends [mccl]`.

## Build InfiniLM

```bash
cd "$WORK"
xmake f -c -y -m release --cxx11-abi=1
xmake build -j16 infinilm_runtime
xmake install infinilm_runtime
xmake build -j16 _infinilm
xmake install _infinilm
python -m pip install janus==2.0.0
```

Install `infinilm_runtime` after its final build and before `_infinilm`. If RT
or Ops changes later, rebuild/install both InfiniLM targets again; a stale
`python/infinilm/_lib/libinfinilm_runtime.so` can retain CPU count zero.

Import and device-count gate:

```bash
export PYTHONPATH=$WORK/python
export LD_LIBRARY_PATH="$WORK/python/infinilm/_lib:$PREFIX/lib:/opt/conda/lib/python3.10/site-packages:/opt/conda/lib/python3.10/site-packages/torch/lib:/opt/maca/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib"
python - <<'PY'
import infinilm, _infinilm
import infinilm._core as core
print(infinilm.__file__, _infinilm.__file__)
print('cpu', core.get_device_count('cpu'))
print('metax', core.get_device_count('metax'))
assert core.get_device_count('cpu') == 1
assert core.get_device_count('metax') >= 1
PY
```

`ldd "$WORK/python/_infinilm.cpython-310-x86_64-linux-gnu.so"` must not report
`not found`.

## Download the model

```bash
export HF_HOME=/data/huggingface_home
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p /data/models /data/huggingface_home
huggingface-cli download Qwen/Qwen3-0.6B \
  --local-dir /data/models/Qwen3-0.6B
```

The validated `model.safetensors` size is about 1.50 GiB.

## Run the inference gate

Check `mx-smi` immediately before running and do not share the GPU with another
process.

```bash
cd /data/InfiniLM
export MACA_PATH=/opt/maca
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/data/InfiniLM/python
export INFINI_ROOT=/data/InfiniLM/prefix
export INFINILM_CXX11_ABI=1
export INFINI_OPS_TRACE_CALLS=0
export LD_LIBRARY_PATH="/data/InfiniLM/python/infinilm/_lib:$INFINI_ROOT/lib:/opt/conda/lib/python3.10/site-packages:/opt/conda/lib/python3.10/site-packages/torch/lib:/opt/maca/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib"
python -c 'import os,runpy,sys; sys.argv=["examples/test_infer.py","--device=metax","--model=/data/models/Qwen3-0.6B","--backend=cpp","--prompt=Hello","--attn=default","--max-new-tokens=16"]; runpy.run_path(sys.argv[0],run_name="__main__"); sys.stdout.flush(); sys.stderr.flush(); os._exit(0)'
```

Success criteria:

```text
Using Static KV Cache with max_cache_len=4096
load weights over!
coherent response beginning with <think>
total_time around one second
```

The `runpy`/`os._exit(0)` pattern flushes output and bypasses extension cleanup
after generation. Check `mx-smi` afterward: no Python process or GPU allocation
should remain.

For operator evidence, set `INFINI_OPS_TRACE_CALLS=1` and repeat with eight
tokens. The successful trace showed MetaX slot 0 calls for Embedding, RmsNorm,
Gemm, RotaryEmbedding, Copy, Fill, Tril, Triu, Add, Softmax,
FusedAddRmsNorm, and SiluAndMul.

## Troubleshooting

- `device index is unavailable`: run the import gate. CPU count zero means RT
  was MetaX-only or staged InfiniLM runtime is stale; rebuild RT with CPU+MetaX
  and reinstall RT, Ops, runtime, and `_infinilm`.
- Missing operator at runtime: add its native slot 0 provider to the profile and
  rebuild. Do not mix native and Torch linked providers in one MACA 3.5 build.
- `Unexpected branch` or `__hneg`: linked Torch headers were compiled by `mxcc`
  with native device headers; keep linked flash providers public-only.
- Non-greedy sampling failure: expected with public-only top-k/top-p; keep the
  default `top_k=1`.
- Build outputs under `prefix/`, `python/`, and `build/` are untracked artifacts.
  The checked-in reproducibility artifacts are this document and the JSON profile.
