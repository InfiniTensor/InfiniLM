#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
BASE=/refactor/candidates/kimi-reduce-model-3e0d89fe-r3261
LM=$BASE/model-src/InfiniLM
COMMON=/refactor/candidates/kimi-reduce-3e0d89fe-r3226/Common/InfiniLM
SDK=/refactor/install/hygon-kimi-3e0d89fe-r2844
HELPER=$BASE/helpers
export PYTHONPYCACHEPREFIX="/local/infini-pycache-${SLURM_JOB_ID}-${SLURM_STEP_ID}-${SLURM_PROCID}"
ABI=$(python -c 'import torch;print(int(torch.compiled_with_cxx11_abi()))')
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="$HELPER:$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$COMMON/python:/old/offline-deps:${PYTHONPATH:-}" OMP_NUM_THREADS=1
unset LD_PRELOAD
test ! -e "$BASE/libmodern_kimi_reduce_r3261.so"
source "$BASE/compile.sh"
python - <<'PY'
import ctypes,datetime,hashlib,json,os,pathlib,subprocess
import torch,infinilm
from infinilm.lib import _infinilm
r=pathlib.Path('/refactor');base=r/'candidates/kimi-reduce-model-3e0d89fe-r3261';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();s=json.loads((r/'results/kimi-reduce-model-r3261-stage.json').read_text());c=json.loads((r/'results/kimi-reduce-common-r3227-build.json').read_text());p=base/'libmodern_kimi_reduce_r3261.so';lib=ctypes.CDLL(str(p));assert s['ready']and c['completed'];assert sha(pathlib.Path(_infinilm.__file__))==c['native_sha256']
for n,h in s['sources_sha256'].items():assert sha(base/n)==h
for group in['unchanged_GPU_objects','unchanged_GPU_helpers']:
 for n,h in s[group].items():assert sha(base/n)==h
syms=subprocess.check_output(['nm','-D',str(p)],text=True)
for symbol in['modern_kimi_reduce_begin_r3226','modern_kimi_reduce_end_r3226']:assert any(v.split()[-2:]==['U',symbol]for v in syms.splitlines()),symbol
assert not any('allreduce_'in v and len(v.split())>=3 and v.split()[-2]in['T','W']for v in syms.splitlines());assert any('CommunicationGroupC'in v and len(v.split())>=3 and v.split()[-2]=='T'for v in syms.splitlines())
for symbol in['create','forward','partition','rank_logits','destroy','layout']:assert getattr(lib,'modern_kimi_engine_'+symbol+'_r2475')
maps={p:sha(pathlib.Path(p))for p in{v.split()[-1]for v in pathlib.Path('/proc/self/maps').read_text().splitlines()if any(n in v for n in['libinfinilm_runtime.so','_infinilm.cpython','libinfiniops.so','libinfinirt.so','libinfiniccl.so','libkimi_grouped_prefill.so'])}}
sdk=json.loads((r/'results/latest-sdk-r2844-Kimi-build.json').read_text())
for name,h in[('libinfinilm_runtime.so',c['runtime_sha256']),('_infinilm.cpython',c['native_sha256']),('libinfiniops.so',sdk['library_sha256']),('libinfinirt.so',sdk['RT_sha256']),('libinfiniccl.so',sdk['CCL_sha256'])]:assert[v for p,v in maps.items()if name in p]==[h]
assert all(str(base/'helpers')in p for p in maps if'libkimi_grouped_prefill.so'in p)
out=r/'results/kimi-reduce-model-r3262-build.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':True,'step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'pins':s['pins'],'library_sha256':sha(p),'stage_sha256':sha(r/'results/kimi-reduce-model-r3261-stage.json'),'mapped_libraries':maps,'single_Common_requestframe_and_allreduce_symbol_resolution':True,'scope':s['scope']};out.write_text(json.dumps(d,indent=2));print('KIMI_REDUCE_MODEL_R3262_BUILD_COMPLETE',d['library_sha256'],flush=True)
PY
