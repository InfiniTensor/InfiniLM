#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export PYTHONPYCACHEPREFIX="/local/infini-pycache-${SLURM_JOB_ID}-${SLURM_STEP_ID}-${SLURM_PROCID}"
BASE=/refactor/candidates/deep-f64-3e0d89fe-r3445
LM=/refactor/candidates/rebased-3e0d89fe-r2842/Model/Deep/InfiniLM
COMMON=/refactor/candidates/rebased-3e0d89fe-r2842/Common/Deep/InfiniLM
SDK=/refactor/install/hygon-deep-cast64-r3315
python - <<'PREFLIGHT'
import json,pathlib
r=pathlib.Path('/refactor');q=json.loads((r/'results/cast64-contract-r3382-audit.json').read_text());assert q['completed']and q['qualified']and q['new_FP64_related_pairs_qualified']and q['all_old9_fullraw_regressions_exact']
PREFLIGHT
mkdir "$BASE"
ABI=$(python -c 'import torch;print(int(torch.compiled_with_cxx11_abi()))')
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="$TORCH_LIB_DIR:$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="/old/offline-deps:${PYTHONPATH:-}" OMP_NUM_THREADS=1
unset NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS LD_PRELOAD
g++ -std=c++17 -O2 -shared -fPIC -pthread -D_GLIBCXX_USE_CXX11_ABI="$ABI" -I"$LM/csrc" -I"$SDK/include" -I"$SDK/include/infiniccl" -I"$COMMON/third_party/json/include" -I"$COMMON/third_party/spdlog/include" -I/opt/dtk/cuda/cuda/include /refactor/incoming/deep_collective_f64_r3445.cpp -L"$COMMON/python" -l:_infinilm.cpython-310-x86_64-linux-gnu.so -L"$COMMON/python/infinilm/_lib" -linfinilm_runtime -L"$SDK/lib" -linfiniccl -linfinirt -linfiniops -L/opt/dtk/cuda/cuda/lib64 -lcudart -Wl,--no-undefined -o "$BASE/libdeep_f64_r3445.so"
python - <<'PY'
import ctypes,datetime,hashlib,json,pathlib
r=pathlib.Path('/refactor');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();p=r/'candidates/deep-f64-3e0d89fe-r3445/libdeep_f64_r3445.so';lib=ctypes.CDLL(str(p));assert lib.deep_projection_r3445_allreduce;sdk=json.loads((r/'results/latest-sdk-r2844-Deep-build.json').read_text());common=json.loads((r/'results/common-latest-r2845-Deep-build.json').read_text());cast=json.loads((r/'results/cast64-sdk-r3375-Deep-load-build.json').read_text());maps={x.split()[-1]:sha(pathlib.Path(x.split()[-1]))for x in pathlib.Path('/proc/self/maps').read_text().splitlines()if any(n in x for n in ['libinfinilm_runtime.so','libinfiniccl.so','libinfinirt.so','libinfiniops.so'])}
for name,h in [('libinfinilm_runtime.so',common['runtime_sha256']),('libinfiniccl.so',sdk['CCL_sha256']),('libinfinirt.so',sdk['RT_sha256']),('libinfiniops.so',cast['library_sha256'])]:assert[v for k,v in maps.items()if name in k]==[h]
out=r/'results/deep-f64-r3445-build.json';assert not out.exists();out.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':True,'library_sha256':sha(p),'mapped_libraries':maps,'source_sha256':{n:sha(r/'incoming'/n)for n in ['deep_collective_f64_r3445.cpp','probe_deep_f64_component_r3445.py']},'scope':'IsolatedCast64SDKqualified3382/Common2845/unchangedCCL6f CastFP32/F64/nativeCCL wrapper, owner-safe error drains; syntheticFP32partials only, noactualremainingmodelrootcause/fullmodelorTPSclaim.'},indent=2))
PY
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
export INFINILM_DENSE_PREFILL_F32_R2475=0
timeout --kill-after=20s 1200 python -u /refactor/incoming/probe_deep_f64_component_r3445.py
