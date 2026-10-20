#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
BASE=/refactor/candidates/glm-batch-model-r3072
SDK=/refactor/install/hygon-glm-3e0d89fe-r2844
LM=/refactor/candidates/rebased-3e0d89fe-r2842/Model/GLM/InfiniLM
COMMON=/refactor/candidates/rebased-3e0d89fe-r2842/Common/GLM/InfiniLM
HELPER=$BASE/helpers
TORCH_LIB_DIR=$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')
ABI=$(python -c 'import torch;print(int(torch.compiled_with_cxx11_abi()))')
export LD_LIBRARY_PATH="$HELPER:$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$COMMON/python:/old/offline-deps:${PYTHONPATH:-}" OMP_NUM_THREADS=1
unset LD_PRELOAD
test ! -e "$BASE/libmodern_glm_batch_r3072.so"
g++ -std=c++17 -O2 /refactor/incoming/glm_batch_geometry_test_r3070.cpp -I"$BASE/src" -o "$BASE/geometry_test"
"$BASE/geometry_test" > "$BASE/geometry-test.json"
g++ -std=c++17 -O2 -shared -fPIC -pthread -D_GLIBCXX_USE_CXX11_ABI="$ABI" -I"$HELPER/src" "$HELPER/src/glm_fp8_cache_r807.cpp" -L"$HELPER" -lglm_fp8_batched_r776 -Wl,-rpath,"$HELPER" -Wl,--no-undefined -o "$HELPER/libglm_fp8_cache_r807.so"
source "$BASE/compile.sh"
python - <<'PY'
import ctypes,datetime,hashlib,json,os,pathlib,subprocess
import torch,infinilm
r=pathlib.Path('/refactor');base=r/'candidates/glm-batch-model-r3072';parent=r/'candidates/models-3e0d89fe-r2856/GLM';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();stage=json.loads((r/'results/glm-batch-model-r3072-stage.json').read_text());assert stage['ready'];test=json.loads((base/'geometry-test.json').read_text());assert test['passed']and test['cases']==50
for name,h in stage['source_sha256'].items():assert sha(base/name)==h
proof=json.loads((r/'results/model-latest-r2856-GLM-build.json').read_text());assert sha(parent/'libmodern_glm_latest_r2856.so')==proof['library_sha256']
for name,h in proof['fresh_GPU_objects'].items():assert sha(base/name)==h
for name,h in proof['fresh_helpers'].items():
 if name!='helpers/libglm_fp8_cache_r807.so':assert sha(base/name)==h
path=base/'libmodern_glm_batch_r3072.so';lib=ctypes.CDLL(str(path));policy=lib.modern_glm_model_batch_cache_policy_r3068;policy.argtypes=[ctypes.c_void_p,ctypes.c_int];stats=lib.modern_glm_model_batch_cache_stats_r3068;stats.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
assert policy(None,0)!=0 and policy(None,2)!=0;values=(ctypes.c_uint64*5)(77,77,77,77,77);assert stats(None,values,5)!=0 and list(values)==[77]*5
def symbols(p):return set(v.split()[-1]for v in subprocess.check_output(['nm','-D','--defined-only',str(p)],text=True).splitlines()if v.strip())
assert symbols(parent/'libmodern_glm_latest_r2856.so')<=symbols(path)
maps={p:sha(pathlib.Path(p))for p in{a.split()[-1]for a in pathlib.Path('/proc/self/maps').read_text().splitlines()if any(s in a for s in['libinfiniops.so','libinfinirt.so','libinfiniccl.so','libinfinilm_runtime.so','libglm_fp8_'])}}
sdk=json.loads((r/'results/latest-sdk-r2844-GLM-build.json').read_text());common=json.loads((r/'results/common-latest-r2845-GLM-build.json').read_text())
for name,h in [('libinfiniops.so',sdk['library_sha256']),('libinfinirt.so',sdk['RT_sha256']),('libinfiniccl.so',sdk['CCL_sha256']),('libinfinilm_runtime.so',common['runtime_sha256'])]:assert[v for p,v in maps.items()if name in p]==[h]
assert all(str(base/'helpers')in p for p in maps if'libglm_fp8_'in p)
out=r/'results/glm-batch-model-r3073-build.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':True,'step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'pins':stage['pins'],'library':str(path),'library_sha256':sha(path),'source_sha256':stage['source_sha256'],'unchanged_current_GPU_objects':proof['fresh_GPU_objects'],'helpers':{p.relative_to(base).as_posix():sha(p)for p in(base/'helpers').glob('*.so')},'mapped_libraries':maps,'CPU_geometry_test':test,'all_parent_dynamic_symbols_retained':True,'null_context_guard_outputs_untouched':True,'scope':'Host adapter/cache library rebuilt over exact current2856 math objects/helper binaries; original currentmodel preserved. DefaultOFF reset/owner/graph policy and actualM32/M64 dispatch stats. Build/load/CPUadmission/nullguard only; fullmodel numerical/owner/perf qualification pending.'};out.write_text(json.dumps(d,indent=2));print(json.dumps({'completed':True,'step':d['step'],'library_sha256':d['library_sha256']}),flush=True)
PY
