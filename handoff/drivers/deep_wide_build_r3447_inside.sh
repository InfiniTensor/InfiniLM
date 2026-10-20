#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
BASE=/refactor/candidates/deep-wide-3e0d89fe-r3443
LM=/refactor/candidates/rebased-3e0d89fe-r2842/Model/Deep/InfiniLM
COMMON=/refactor/candidates/rebased-3e0d89fe-r2842/Common/Deep/InfiniLM
SDK=/refactor/install/hygon-deep-cast64-r3315
HELPER=/refactor/candidates/models-3e0d89fe-r2856/Deep/helpers
ABI=$(python -c 'import torch;print(int(torch.compiled_with_cxx11_abi()))')
TORCH_LIB_DIR=$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')
export LD_LIBRARY_PATH="$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$COMMON/python:/old/offline-deps:${PYTHONPATH:-}" OMP_NUM_THREADS=1
unset LD_PRELOAD
test ! -e "$BASE/libmodern_deep_wide_r3443.so"
source "$BASE/compile.sh"
python - <<'PY'
import ctypes,datetime,hashlib,json,pathlib
import torch,infinilm
r=pathlib.Path('/refactor');base=r/'candidates/deep-wide-3e0d89fe-r3443';stage=json.loads((r/'results/deep-wide-r3443-stage.json').read_text());assert stage['ready'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n,h in stage['files_sha256'].items():assert sha(base/n)==h,n
p=base/'libmodern_deep_wide_r3443.so';lib=ctypes.CDLL(str(p));f=lib.modern_deep_precision_policy_r3443;f.argtypes=[ctypes.c_int];g=lib.modern_deep_precision_stats_r3443;g.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
assert f(-1)==1 and f(5)==1 and f(99)==1 and f(0)==0
for i in range(8):
 a=(ctypes.c_uint64*6)(*[777]*6);assert g(i,a,6)==0 and list(a)==[0]*6
for dev,n in [(-1,6),(8,6),(0,5),(0,7)]:
 a=(ctypes.c_uint64*8)(*[777]*8);assert g(dev,a,n)==1 and list(a)==[777]*8
assert g(0,None,6)==1
for mode in range(5):assert f(mode)==0
assert f(0)==0

maps={s.split()[-1]:sha(pathlib.Path(s.split()[-1]))for s in pathlib.Path('/proc/self/maps').read_text().splitlines()if any(n in s for n in ['libinfiniops.so','libinfinirt.so','libinfiniccl.so','libinfinilm_runtime.so'])};sdk=json.loads((r/'results/latest-sdk-r2844-Deep-build.json').read_text());common=json.loads((r/'results/common-latest-r2845-Deep-build.json').read_text())
cast=json.loads((r/'results/cast64-sdk-r3375-Deep-load-build.json').read_text());assert cast['completed']
for name,h in [('libinfiniops.so',cast['library_sha256']),('libinfinirt.so',sdk['RT_sha256']),('libinfiniccl.so',sdk['CCL_sha256']),('libinfinilm_runtime.so',common['runtime_sha256'])]:assert[v for k,v in maps.items()if name in k]==[h]
out=r/'results/deep-wide-r3447-build.json';assert not out.exists();out.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':True,'pins':stage['pins'],'library_sha256':sha(p),'stage_sha256':sha(r/'results/deep-wide-r3443-stage.json'),'mapped_libraries':maps,'scope':'Host-only isolated build/import/maps/invaliduntouched guards and defaultOFF; modes1native-sync control,2attentionF64,3addsFFN,4addsindexer. No component/fullmodel/performance qualification.'},indent=2));print('DEEP_WIDE_R3447_BUILD_COMPLETE',flush=True)
PY
