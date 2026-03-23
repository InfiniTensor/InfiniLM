# MetaX Flash-Attn Build & Run Guide (InfiniCore + InfiniLM)

This guide documents how to build **InfiniCore** + **InfiniLM** for **MetaX / hpcc** with **Flash-Attn**, run `jiuge.py`, and start the **HTTP inference server** with `--attn=flash-attn` (optionally with `--enable-graph`).

Assumptions:
- You run everything inside Docker container `dev2`.
- Repo root is mounted at: `/home/zenghua/workspace/fla-support`
- Model dir is mounted at: `/data-aisoft/zenghua/models/9g_8b_thinking_llama` (adjust as needed)

## 0. Container + environment

Use the workflow checklist for GPU/env basics:
- Set GPU: `export HPCC_VISIBLE_DEVICES=N`
- Set `PYTHONPATH` so InfiniCore/InfiniLM python modules are importable.
- Ensure `LD_LIBRARY_PATH` contains InfiniCore libs.

Example (inside `dev2`):

```bash
export REPO=/home/zenghua/workspace/fla-support
export PYTHONPATH=$REPO/InfiniLM/python:$REPO/InfiniCore/python:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/root/.infini/lib:${LD_LIBRARY_PATH:-}
export HPCC_VISIBLE_DEVICES=0
```

## 1. Prereqs: Flash-Attn .so must exist

InfiniCore links the pip-provided Flash-Attn shared library (MetaX path).

Verify the file exists (dev2 conda site-packages path):

```bash
ls -la /opt/conda/lib/python3.10/site-packages/flash_attn_2_cuda*.so
```

If you need to override the exact `.so` path, export:

```bash
export FLASH_ATTN_2_CUDA_SO=/opt/conda/lib/python3.10/site-packages/flash_attn_2_cuda*.so
```

## 2. Build InfiniCore (MetaX + Flash-Attn + ATen)

Flash-attn integration is enabled only when `xmake` config option `flash-attn` is set (non-empty) **and** you enable `metax-gpu` + `aten`.

Build inside `dev2`:

```bash
cd "$REPO/InfiniCore"

# (Optional) Make flash-attn .so path explicit so MetaX linking is deterministic.
# If you already exported `FLASH_ATTN_2_CUDA_SO` in your shell, you can skip these.
export FLASH_ATTN_2_CUDA_SO=/opt/conda/lib/python3.10/site-packages/flash_attn_2_cuda*.so

# (Optional) If you want to override the "expected container path" fallback used by xmake,
# you can also export:
# export FLASH_ATTN_METAX_CUDA_SO_CONTAINER=/opt/conda/lib/python3.10/site-packages/flash_attn_2_cuda.cpython-310-aarch64-linux-gnu.so

# Clean rebuild is recommended when changing flash-attn/aten settings.
# This uses the helper install script, which calls `xmake f ... -cv` and then builds.
python scripts/install.py --metax-gpu=y --aten=y --flash-attn=.

# Install Python/C++ libs into $INFINI_ROOT (default: ~/.infini)
xmake install infiniop-test >/dev/null 2>&1 || true
```

Notes:
- `--flash-attn=.` is sufficient to activate the flash-attn build/link logic in `xmake.lua`.
- Actual linking is done using `FLASH_ATTN_2_CUDA_SO` (or the hardcoded expected pip `.so` path in `xmake.lua` for `dev2`).

## 3. Build InfiniLM

- `_infinilm` (python extension module)

```bash
cd "$REPO/InfiniLM"

# 1) Install python package (editable)
pip install -e .

# 2) Build + install libraries
xmake build  _infinilm
xmake install  _infinilm
```

## 4. Run `jiuge.py` (MetaX)

Use the test script to validate the MetaX execution path:

```bash
python "$REPO/InfiniLM/scripts/jiuge.py" --metax /data-aisoft/zenghua/models/9g_8b_thinking_llama 1 --verbose
```

Notes:
- `jiuge.py` selects the device (`--metax`) but does not expose an `--attn` flag.
- If you built InfiniCore with Flash-Attn enabled, the server/runtime should be able to use the Flash-Attn kernels. `jiuge.py` mainly checks correctness of the MetaX pipeline.

## 5. Launch inference server with `--attn=flash-attn`

Start the OpenAI-compatible HTTP server:

```bash
python -m infinilm.server.inference_server \
  --metax \
  --model_path=/data-aisoft/zenghua/models/9g_8b_thinking_llama \
  --attn=flash-attn \
  # Enable graph compiling/capture (recommended for the graph-mode flash-attn validation)
  --enable-graph \
  --host=0.0.0.0 --port=8000 \
  --max_tokens=2048 \
  --cache_type paged \
  --max_batch_size=4 \
  --num_blocks=512 \
  --block_size=256
```

## 6. Verify server

Health + model list:

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

Simple generation test:

```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a 1-sentence summary of attention.",
    "max_tokens": 64,
    "temperature": 0.0
  }'
```

## 7. Troubleshooting

1. Server starts but Flash-Attn is not used
   - Ensure InfiniCore was built with: `--aten=y --flash-attn=.` and `--metax-gpu=y`.
   - Ensure `FLASH_ATTN_2_CUDA_SO` points to an existing `flash_attn_2_cuda*.so` (or the hardcoded expected path exists in `dev2`).

2. `libinfinicore_cpp_api.so: cannot open shared object file`
   - Export InfiniCore lib path into `LD_LIBRARY_PATH` (typically `/root/.infini/lib`).

3. `jiuge.py` fails on MetaX
   - Rebuild both InfiniCore + InfiniLM after changing flash-attn/aten.
   - Confirm model directory contains `config.json` + weight files that `transformers` can load.
