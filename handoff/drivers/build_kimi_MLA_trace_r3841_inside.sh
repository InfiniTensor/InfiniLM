#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export PYTHONPYCACHEPREFIX="/local/spec-build-${SLURM_JOB_ID}-${SLURM_STEP_ID}-${SLURM_PROCID}"
BASE=/refactor/candidates/kimi-MLA-trace-5f91ad7-r3841
LM=$BASE/InfiniLM
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM
SDK=/refactor/install/hygon-kimi-5f91ad7-r3672
HELPER=/refactor/candidates/models-5f91ad7-r3675/Kimi/helpers
ABI=$(python -c 'import torch;print(int(torch.compiled_with_cxx11_abi()))')
TORCH_LIB_DIR=$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')
export LD_LIBRARY_PATH="$HELPER:$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$COMMON/python:/old/offline-deps:${PYTHONPATH:-}" OMP_NUM_THREADS=1
unset LD_PRELOAD
test ! -e "$BASE/libmodern_kimi_MLA_trace_r3841.so"
OPS=/refactor/candidates/rebased-5f91ad7-r3668/Ops/Kimi/InfiniOps
for variant in source_first installed_only; do
 EXTRA=()
 if [ "$variant" = source_first ]; then EXTRA=(-I"$OPS/src"); fi
 g++ -std=c++17 -O0 -D_GLIBCXX_USE_CXX11_ABI="$ABI" "${EXTRA[@]}" -I"$SDK/include" -I"$COMMON/third_party/json/include" -I/opt/dtk/cuda/cuda/include /refactor/incoming/kimi_registry_header_r3712.cpp -L"$SDK/lib" -linfiniops -linfinirt -o "$BASE/registry-$variant"
 "$BASE/registry-$variant" > "$BASE/registry-$variant.json"
done
python - <<'PROOF'
import json,pathlib
b=pathlib.Path('/refactor/candidates/kimi-MLA-trace-5f91ad7-r3841');a=json.loads((b/'registry-source_first.json').read_text());z=json.loads((b/'registry-installed_only.json').read_text());assert all(z[k]for k in z),('installed_registry_missing',z);print('REGISTRY_HEADER_COMPARISON',json.dumps({'source_first':a,'installed_only':z}),flush=True)
PROOF
source "$BASE/compile.sh"
python - <<'PY'
import ctypes,datetime,hashlib,json,pathlib,os
import torch,infinilm
r=pathlib.Path('/refactor');base=r/'candidates/kimi-MLA-trace-5f91ad7-r3841';stage=json.loads((r/'results/kimi-MLA-trace-r3841-stage.json').read_text());assert stage['ready'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n,h in stage['files_sha256'].items():assert sha(base/n)==h,n
path=base/'libmodern_kimi_MLA_trace_r3841.so';lib=ctypes.CDLL(str(path));lib.modern_kimi_engine_error_r2475.restype=ctypes.c_char_p
os.environ['INFINILM_KIMI_SPEC_R3632']='1';assert lib.modern_kimi_spec_contract_r3632()==0,lib.modern_kimi_engine_error_r2475();lib.modern_kimi_spec_contract_report_r3632.restype=ctypes.c_char_p;contract=json.loads(lib.modern_kimi_spec_contract_report_r3632());assert contract['completed'];del os.environ['INFINILM_KIMI_SPEC_R3632']
assert sha(base/'spec_kda.o')==stage['spec_kda_object_sha256']
assert lib.modern_kimi_spec_math_counts_r3676 and lib.modern_kimi_spec_snapshot_counts_r3775
assert lib.modern_kimi_engine_spec_r3632 and lib.modern_kimi_engine_create_spec_r3632 and lib.modern_owner_affinity_report_r2392
maps={s.split()[-1]:sha(pathlib.Path(s.split()[-1]))for s in pathlib.Path('/proc/self/maps').read_text().splitlines()if any(n in s for n in ['libinfinilm_runtime.so','libinfiniops.so','libinfinirt.so','libinfiniccl.so'])};sdk=json.loads((r/'results/latest-sdk-r3672-Kimi-build.json').read_text());common=json.loads((r/'results/common-latest-r3673-Kimi-build.json').read_text())
for n,h in [('libinfiniops.so',sdk['library_sha256']),('libinfinirt.so',sdk['RT_sha256']),('libinfiniccl.so',sdk['CCL_sha256']),('libinfinilm_runtime.so',common['runtime_sha256'])]:assert[v for k,v in maps.items()if n in k]==[h]
out=r/'results/kimi-MLA-trace-r3841-build.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':True,'pins':stage['pins'],'library_sha256':sha(path),'stage_sha256':sha(r/'results/kimi-MLA-trace-r3841-stage.json'),'mapped_libraries':maps,'registry_header_comparison':{v:json.loads((base/('registry-'+v+'.json')).read_text())for v in['source_first','installed_only']},'CPU_contract':contract,'GPU_objects_unchanged_latest3586':stage['fresh_latest_GPU_objects'],'scope':'Isolated opt-in multi-batch speculation verifier build/CPU contracts only. Actual full-model allposition parity, rejection replay, complete EOS sequences and measured speedup remain required. Excluded from main optimization metric.'};out.write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
PY
