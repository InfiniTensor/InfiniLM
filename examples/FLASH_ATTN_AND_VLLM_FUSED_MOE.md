# Flash-attn + fused MoE (clean guide)

This guide is for running **MiniCPM5 MoE** with:

- **FlashAttention-2** (`--attn flash-attn` + paged KV)
- **Fused MoE** (vendored vLLM-derived Triton kernels under `infinicore.vendor.vllm_fused_moe`, `torch.ops.infinilm.*`; **no `pip install vllm`** for this path)

Keep **one Python / one torch** end-to-end: the same interpreter must run `jiuge.py`, satisfy **Triton** + fused vendor imports, and build InfiniCore with `--aten=y`.

## Requirements (TL;DR)

- **Python venv**: `$REPO/.venv-vllm` (name kept for flash-attn workflows)
- **Triton**: via `InfiniLM/examples/requirements-vllm-fused-moe.txt` (and typically your CUDA torch wheel)
- **setuptools**: `setuptools>=77.0.3,<81.0.0` (recommended bound on Python 3.12 images)
- **flash-attn wheel** must match **torch major.minor**
  - For **torch 2.10 + py312 + cxx11abiTRUE**: use **flash-attn 2.8.1** `cu12torch2.10` wheel (below)
- **Run flags**: `--attn flash-attn --enable-paged-attn` (flash decode expects paged KV)

## Install (TUNA + flash-attn wheel)

```bash
REPO=/home/zenghua/workspace/minicpm5-moe-support
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
cd "$REPO"
python3 -m venv .venv-vllm
source .venv-vllm/bin/activate
pip install -U pip wheel
pip install -r InfiniLM/examples/requirements-vllm-fused-moe.txt
```

Install **flash-attn** from GitHub release (PyPI often only has sdist; building fails when image `nvcc` != torch CUDA):

```bash
pip install --no-cache-dir \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl' \
  -i "${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
```

If GitHub is slow, download on the **host** with proxy `127.0.0.1:57890` and copy into the container:

```bash
export http_proxy=http://127.0.0.1:57890
export https_proxy=http://127.0.0.1:57890
curl -fL -o /tmp/flash_attn.whl \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
docker cp /tmp/flash_attn.whl minicpm5-moe:/tmp/flash_attn-2.8.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

Then inside container (venv activated):

```bash
pip install /tmp/flash_attn-2.8.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Sanity check (in venv):

```bash
python -c "import flash_attn, torch; import glob, os; sp=os.path.dirname(os.path.dirname(torch.__file__)); print('flash_attn', flash_attn.__version__); print(glob.glob(sp+'/flash_attn_2_cuda*.so'))"
```

## Build (venv)

```bash
export XMAKE_ROOT=y
export PATH="$HOME/.local/bin:$PATH"
REPO=/home/zenghua/workspace/minicpm5-moe-support
source "$REPO/.venv-vllm/bin/activate"
export PYTHONPATH=$REPO/InfiniLM/python:$REPO/InfiniCore/python

cd "$REPO/InfiniCore"
python scripts/install.py --nv-gpu=y --cuda_arch=sm_80 --aten=y
xmake build _infinicore && xmake install _infinicore

cd "$REPO/InfiniLM"
xmake build _infinilm && xmake install _infinilm
```

## Run (validated)

### Fast validation (original checkpoint directory)

```bash
docker exec minicpm5-moe bash -lc '
set -euo pipefail
REPO=/home/zenghua/workspace/minicpm5-moe-support
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
source "$REPO/.venv-vllm/bin/activate"
export PYTHONPATH=$REPO/InfiniLM/python:$REPO/InfiniCore/python
TORCH_LIB=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), \"lib\"))")
unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/root/.infini/lib:$TORCH_LIB:$REPO/.venv-vllm/lib/python3.12/site-packages:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu
cd "$REPO/InfiniLM/examples"
export INFINILM_FORCE_MOE_BACKEND=vllm_fused
python -u jiuge.py --nvidia --model-path /data-aisoft/zenghua/models/minicpm5.16a3.v0314 \
  --prompt "Hi" --max-new-tokens 8 --top-k 1 \
  --attn flash-attn --enable-paged-attn --paged-kv-block-size 256
'
```

### Full “one-liner” (rebuild + flash-attn + fused MoE)

```bash
docker exec minicpm5-moe bash -lc 'set -euo pipefail; export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}; REPO=/home/zenghua/workspace/minicpm5-moe-support; MODEL=/data-aisoft/zenghua/models/minicpm5.16a3.v0314; export XMAKE_ROOT=y; source "$REPO/.venv-vllm/bin/activate"; export PATH=$REPO/.venv-vllm/bin:$HOME/.local/bin:$PATH; export PYTHONPATH=$REPO/InfiniLM/python:$REPO/InfiniCore/python; cd $REPO/InfiniCore && python scripts/install.py --nv-gpu=y --cuda_arch=sm_80 --aten=y && xmake build _infinicore && xmake install _infinicore && cd $REPO/InfiniLM && xmake build _infinilm && xmake install _infinilm && python -c "import flash_attn" >/dev/null 2>&1 || pip install --no-build-isolation flash-attn && TORCH_LIB=$(python -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__),\"lib\"))") && unset LD_LIBRARY_PATH && export LD_LIBRARY_PATH=/root/.infini/lib:$TORCH_LIB:$REPO/.venv-vllm/lib/python3.12/site-packages:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu && cd $REPO/InfiniLM/examples && export INFINILM_FORCE_MOE_BACKEND=vllm_fused && unset INFINILM_DISABLE_VLLM_FUSED_MOE && python -u jiuge.py --nvidia --model-path "$MODEL" --prompt "Hi" --max-new-tokens 24 --top-k 1 --attn flash-attn --enable-paged-attn --paged-kv-block-size 256'
```

Expected log:

- Contains: **`[vllm_fused_moe] preflight`** (may warn “no tuned JSON, using defaults”)
- Does **not** contain: `FlashAttentionImpl decode failed` / “Tensor that doesn't have storage”

## Troubleshooting (common)

- **flash-attn build fails with CUDA mismatch**: don’t build from sdist; install the matching GitHub wheel (`cu12torch2.10`, correct `cxx11abi`).
- **decode says “Tensor that doesn't have storage”**: you preloaded a system flash `.so` against venv torch; install flash-attn in the venv and avoid `LD_PRELOAD` of `/usr/local/...` for venv runs.
- **OOM during vLLM preflight**: set `INFINILM_SKIP_VLLM_FUSED_MOE_PREFLIGHT=1` (or cap warmup bytes via `INFINILM_VLLM_FUSED_WARMUP_MAX_BYTES`).

