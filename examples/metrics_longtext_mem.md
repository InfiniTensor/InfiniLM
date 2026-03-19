### MiniCPM-SALA long-context metrics + memory history

**Goal**: record reproducible long-context runs with:
- **time** (prefill TTFT / throughput)
- **peak GPU memory** (from 1s `nvidia-smi` polling)
- exact **command lines** and key env

**Notes**
- All commands are intended to run **inside** docker container `minicpm-sala`.
- Prefer using an idle GPU; set `CUDA_VISIBLE_DEVICES=<N>` and poll the **physical** GPU `<N>` with `nvidia-smi -i <N> ...`.
- For InfiniLM + InfLLM-v2 builds, `libinfinicore_cpp_api.so` may require preloading `infllm_v2` with `RTLD_GLOBAL` before importing `infinicore`.

---

## History table

| date | backend | target_input_tokens | max_new_tokens | cache_mode | peak_mem_mib | total_time_ms | prefill_ttft_ms | prefill_throughput_tok_s | gpu |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| 2026-03-18 | hf | 16384 | 1 | — | 38091 | 1757.21 | — | 9325.01 | 2 |
| 2026-03-18 | infinilm(v0) | 16384 | 1 | static_fit | 43111 | 2771.90 | 2771.56 | 5912.19 | 2 |
| 2026-03-18 | infinilm(latest) | 16384 | 1 | static_fit | 40811 | 2833.95 | 2833.75 | 5782.45 | 1 |
| 2026-03-18 | hf | 32768 | 1 | — | 41173 | 3537.65 | — | 9263.22 | 2 |
| 2026-03-18 | infinilm(v0) | 32768 | 1 | static_fit | 65237 | 18570.43 | 18569.97 | 1764.68 | 2 |
| 2026-03-18 | infinilm(latest) | 32768 | 1 | static_fit | 65233 | 5971.07 | 5970.86 | 5488.32 | 1 |
| 2026-03-18 | hf | 524288 | 1 | — | 59591 | OOM | — | — | 3 |
| 2026-03-18 | hf | 65536 | 1 | — | 47319 | 7340.99 | — | 8927.67 | 1 |
| 2026-03-18 | hf | 131072 | 1 | — | 61641 | 15290.39 | — | 8572.31 | 1 |
| 2026-03-18 | hf | 262144 | 1 | — | 80059 | OOM | — | — | 1 |

---

## Commands (repro)

### Logits correctness gate (HF vs InfiniLM)

Run (inside `minicpm-sala`) to sanity-check HF vs InfiniLM prefill logits on a short prompt:

```bash
REPO=/workspace
MODEL=/data-aisoft/zenghua/models/OpenBMB/MiniCPM-SALA
export CUDA_VISIBLE_DEVICES=1
export HF_CUDA_INDEX=0
export INFINILM_CUDA_INDEX=0
export PYTHONPATH=$REPO/InfiniLM/python:$REPO/InfiniCore/python:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/root/.infini/lib:${LD_LIBRARY_PATH:-}
cd $REPO/InfiniLM/examples

python3 minicpm_sala_logits_sanity.py \
  --model_path "$MODEL" \
  --mode prefill \
  --prompt "How are you? Tell me a short joke." \
  --k 10
```

Recorded output (2026-03-18, GPU=1):

```text
SANITY_ONELINE ratio=0.9889 max_diff=0.1875 mean_diff=0.0682
```

### GPU scan

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits
```

### HF-only prefill (32k) with 1s memory polling

```bash
REPO=/workspace
MODEL=/data-aisoft/zenghua/models/OpenBMB/MiniCPM-SALA
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=$REPO/InfiniLM/python:$REPO/InfiniCore/python:${PYTHONPATH:-}
cd $REPO/InfiniLM/examples

python3 compare_inference_speed.py \
  --model_path "$MODEL" \
  --target_input_tokens 32768 \
  --max_new_tokens 1 \
  --backends hf \
  --hf_mode forward_prefill \
  --hf_forward_use_cache \
  --hf_forward_warmup 1 \
  --hf_forward_iters 1 \
  --hf_attn_implementation flash_attention_2 \
  & pid=$!

echo "[mem] polling physical GPU 2 while pid=$pid"
while kill -0 $pid 2>/dev/null; do
  date +"%F %T"
  nvidia-smi -i 2 --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
  sleep 1
done
wait $pid
```

### InfiniLM-only (32k) with InfLLM-v2 preload + 1s memory polling

```bash
REPO=/workspace
MODEL=/data-aisoft/zenghua/models/OpenBMB/MiniCPM-SALA
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=$REPO/InfiniLM/python:$REPO/InfiniCore/python:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/root/.infini/lib:$REPO/InfiniLM/build/linux/x86_64/release:${LD_LIBRARY_PATH:-}
cd $REPO/InfiniLM/examples

python3 - <<'PY' & pid=$!
import ctypes, os, runpy, sys
ctypes.CDLL("/usr/local/lib/python3.12/dist-packages/infllm_v2/C.cpython-312-x86_64-linux-gnu.so", mode=ctypes.RTLD_GLOBAL)
sys.argv = [
  "compare_inference_speed.py",
  "--model_path", os.environ["MODEL"],
  "--target_input_tokens", "32768",
  "--max_new_tokens", "1",
  "--backends", "infinilm",
  "--no_hf",
  "--infinilm_inprocess",
  "--infinilm_cache_mode", "static_fit",
]
runpy.run_path("compare_inference_speed.py", run_name="__main__")
PY

echo "[mem] polling physical GPU 2 while pid=$pid"
while kill -0 $pid 2>/dev/null; do
  date +"%F %T"
  nvidia-smi -i 2 --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
  sleep 1
done
wait $pid
```
