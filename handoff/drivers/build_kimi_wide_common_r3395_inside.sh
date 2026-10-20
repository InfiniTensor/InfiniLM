#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export PATH=/root/.local/bin:/opt/dtk/llvm/bin:/opt/dtk/bin:$PATH
LM=/refactor/candidates/kimi-wide-3e0d89fe-r3393/Common/InfiniLM
SDK=/refactor/install/hygon-kimi-cast64-r3315
export PYTHONPYCACHEPREFIX="/local/pycache-${SLURM_JOB_ID}-${SLURM_STEP_ID}"
mkdir /local/compiler-tmp
export TMPDIR=/local/compiler-tmp
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')"
export INFINI_ROOT="$SDK" CUDA_HOME=/opt/dtk/cuda/cuda CUDA_PATH=/opt/dtk/cuda/cuda
export INFINILM_CXX11_ABI="$(python -c 'import torch;print(int(torch.compiled_with_cxx11_abi()))')"
export LD_LIBRARY_PATH="$SDK/lib:$TORCH_LIB_DIR:$CUDA_HOME/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export PYBIND11_INCLUDE_DIR=/refactor/build-deps/pybind11-3.0.4/include
PYTHON_INCLUDE_DIR=$(python -c 'import sysconfig;print(sysconfig.get_path("include"))')
export CPLUS_INCLUDE_PATH="$PYTHON_INCLUDE_DIR:${CPLUS_INCLUDE_PATH:-}"
export XMAKE_PROGRAMDIR=/root/.local/share/xmake XMAKE_GLOBALDIR=/root XMAKE_ROOT=y OMP_NUM_THREADS=1
unset LD_PRELOAD INFINILM_DENSE_PREFILL_F32_R2475 INFINILM_MXFP4_TILE_TP_R2559
python - <<'PY'
import hashlib,json,pathlib
r=pathlib.Path('/refactor');root=r/'candidates/kimi-wide-3e0d89fe-r3393/Common/InfiniLM';d=json.loads((r/'results/kimi-wide-r3393-stage.json').read_text());assert d['ready']and d['pins']['InfiniLM']=='3e0d89fe881ca9fb2411c6e35ffc913c7ea6654d'
for n,h in d['csrc_sha256']['Common'].items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
assert not list((root/'python').rglob('*.so'))
f64=json.loads((r/'results/kimi-f64-r3388-audit.json').read_text());assert f64['completed']and f64['cases']==192 and f64['F64_all_cases_CPU_FP64sum_RNE_F32_BF16_and_rows_repeat_exact']
PY
cd "$LM"
xmake f -y -m release --cxx11-abi="$INFINILM_CXX11_ABI" -cv
xmake build -y -j16 _infinilm
xmake install -y infinilm_runtime
xmake install -y _infinilm
export LD_LIBRARY_PATH="$SDK/lib:$LM/python:$LM/python/infinilm/_lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$LM/python:/old/offline-deps:${PYTHONPATH:-}"
python - <<'PY'
import ctypes as C,datetime,hashlib,json,os,pathlib
import torch,infinilm
from infinilm.lib import _infinilm
r=pathlib.Path('/refactor');root=r/'candidates/kimi-wide-3e0d89fe-r3393/Common/InfiniLM';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();stage=json.loads((r/'results/kimi-wide-r3393-stage.json').read_text());native=pathlib.Path(_infinilm.__file__).resolve();assert native.is_relative_to(root/'python')
lib=C.CDLL(str(native));lib.modern_kimi_reduce_policy_r3226.argtypes=[C.c_int];lib.modern_kimi_reduce_stats_r3226.argtypes=[C.c_int,C.POINTER(C.c_uint64),C.c_size_t];lib.modern_kimi_reduce_begin_r3226.argtypes=[C.c_void_p,C.c_size_t];lib.modern_kimi_reduce_end_r3226.argtypes=[C.c_void_p]
lib.modern_kimi_reduce_wide_stats_r3393.argtypes=[C.c_int,C.POINTER(C.c_uint64),C.c_size_t]
for dev in range(8):
 v=(C.c_uint64*5)();w=(C.c_uint64*6)();assert lib.modern_kimi_reduce_stats_r3226(dev,v,5)==0 and list(v)==[0]*5;assert lib.modern_kimi_reduce_wide_stats_r3393(dev,w,6)==0 and list(w)==[0]*6
w=(C.c_uint64*6)(77,77,77,77,77,77);assert lib.modern_kimi_reduce_wide_stats_r3393(8,w,6)!=0 and lib.modern_kimi_reduce_wide_stats_r3393(0,w,5)!=0 and list(w)==[77]*6
assert lib.modern_kimi_reduce_policy_r3226(2)==0 and lib.modern_kimi_reduce_policy_r3226(1)==0 and lib.modern_kimi_reduce_policy_r3226(0)==0
assert lib.modern_kimi_reduce_policy_r3226(99)==1 and lib.modern_kimi_reduce_begin_r3226(None,8)!=0 and lib.modern_kimi_reduce_end_r3226(None)!=0
v=(C.c_uint64*5)(77,77,77,77,77);assert lib.modern_kimi_reduce_stats_r3226(8,v,5)!=0 and lib.modern_kimi_reduce_stats_r3226(0,v,4)!=0 and list(v)==[77]*5
maps={p:sha(pathlib.Path(p))for p in{v.split()[-1]for v in pathlib.Path('/proc/self/maps').read_text().splitlines()if any(k in v for k in['libinfiniops.so','libinfiniccl.so','libinfinirt.so','libinfinilm_runtime.so'])}}
sdk=json.loads((r/'results/latest-sdk-r2844-Kimi-build.json').read_text());cast=json.loads((r/'results/cast64-sdk-r3375-Kimi-load-build.json').read_text());rt=root/'python/infinilm/_lib/libinfinilm_runtime.so'
for name,h in[('libinfiniops.so',cast['library_sha256']),('libinfiniccl.so',sdk['CCL_sha256']),('libinfinirt.so',sdk['RT_sha256'])]:assert[h0 for p,h0 in maps.items()if name in p]==[h]
assert[p for p in maps if'libinfinilm_runtime.so'in p]==[str(rt)]
for n,h in stage['csrc_sha256']['Common'].items():assert sha(root/n)==h
out=r/'results/kimi-wide-common-r3395-build.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':True,'step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'pins':sdk['pins'],'native':str(native),'native_sha256':sha(native),'runtime_sha256':sha(rt),'mapped_libraries':maps,'defaultOFF_all8_stats_zero':True,'invalid_policy_null_frame_and_stats_outputs_untouched':True,'source_stage_sha256':sha(r/'results/kimi-wide-r3393-stage.json'),'scope':'FreshprivateCommon3393 Runtime/Python+Cast64SDKqualified3382, unchangedRT/CCLpins. CAPI3226retained, defaultOFFpolicy2promotesBF16andFP32SUM onlyactualmulti-request; B1unpromoted. Additional6wide counters. Onlybuild/load/invalid/default guards, realowner/CCL/fullmodel numerical+performance qualification pending; sameSDK/mathobjects not changed.'};out.write_text(json.dumps(d,indent=2));print('KIMI_WIDE_COMMON_R3395_BUILD_COMPLETE',d['step'],d['native_sha256'],flush=True)
PY
