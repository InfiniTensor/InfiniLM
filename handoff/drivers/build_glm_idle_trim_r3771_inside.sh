#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export PYTHONPYCACHEPREFIX="/local/latest-model-${SLURM_JOB_ID}-${SLURM_STEP_ID}-${SLURM_PROCID}"
export OMP_NUM_THREADS=1
NAME=$(printf '%s' "$MODEL_VARIANT" | tr '[:upper:]' '[:lower:]')
BASE=/refactor/candidates/glm-idle-trim-5f91ad7-r3771
LM=/refactor/candidates/rebased-5f91ad7-r3668/Model/$MODEL_VARIANT/InfiniLM
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/$MODEL_VARIANT/InfiniLM
OPS=/refactor/candidates/rebased-5f91ad7-r3668/Ops/$MODEL_VARIANT/InfiniOps
SDK=/refactor/install/hygon-${NAME}-5f91ad7-r3672
HELPER=$BASE/helpers
ABI=$(python -c 'import torch;print(int(torch.compiled_with_cxx11_abi()))')
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="$HELPER:$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$COMMON/python:/old/offline-deps:${PYTHONPATH:-}"
unset LD_PRELOAD INFINILM_DENSE_PREFILL_F32_R2475 INFINILM_MXFP4_TILE_TP_R2559
source "$BASE/compile.sh"
python - <<'PY'
import ctypes,datetime,hashlib,json,os,pathlib,subprocess
import torch,infinilm
from infinilm.lib import _infinilm
r=pathlib.Path('/refactor');name=os.environ['MODEL_VARIANT'];base=r/'candidates/glm-idle-trim-5f91ad7-r3771';v=json.loads((r/'results/models-latest-r3675-stage.json').read_text())['variants'][name];common=r/'candidates/rebased-5f91ad7-r3668/Common'/name/'InfiniLM';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();path=base/'libmodern_glm_idle_trim_r3771.so';lib=ctypes.CDLL(str(path))
symbols=v['CAPI_symbols']
if name=='Kimi':symbols=['modern_kimi_engine_'+s+'_r2475'for s in ['create','forward','partition','rank_logits','destroy','layout']]
for s in symbols:assert getattr(lib,s),s
assert lib.modern_glm_idle_trim_r3771
assert lib.modern_owner_affinity_report_r2392
maps={p:sha(pathlib.Path(p))for p in{a.split()[-1]for a in pathlib.Path('/proc/self/maps').read_text().splitlines()if any(s in a for s in ['libinfiniops.so','libinfiniccl.so','libinfinirt.so','libinfinilm_runtime.so','_infinilm.cpython','libkimi_grouped_prefill.so','libglm_fp8_'])}}
sdk=json.loads((r/'results'/f'latest-sdk-r3672-{name}-build.json').read_text());rt=json.loads((r/'results'/f'common-latest-r3673-{name}-build.json').read_text())
assert[h for p,h in maps.items()if'libinfiniops.so'in p]==[sdk['library_sha256']]
assert[h for p,h in maps.items()if'libinfinirt.so'in p]==[sdk['RT_sha256']]
assert[h for p,h in maps.items()if'libinfiniccl.so'in p]==[sdk['CCL_sha256']]
assert[h for p,h in maps.items()if'libinfinilm_runtime.so'in p]==[rt['runtime_sha256']]
assert pathlib.Path(_infinilm.__file__).resolve().is_relative_to(common/'python')
assert all(str(base/'helpers') in p for p in maps if any(s in p for s in ['libkimi_grouped_prefill.so','libglm_fp8_']))
if name=='Kimi':
 layout=json.loads((base/'layout-contract.json').read_text());assert layout['completed'] and layout['layouts']==48 and layout['ranks']==1430
out=r/'results'/'glm-idle-trim-r3771-build.json';assert not out.exists();d={'stage_manifest':json.loads((r/'results/glm-idle-trim-r3771-stage.json').read_text()),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':True,'variant':name,'pins':sdk['pins'],'library':str(path),'library_sha256':sha(path),'compile_sha256':sha(base/'compile.sh'),'fresh_GPU_objects':{p.relative_to(base).as_posix():sha(p)for p in base.rglob('*.o')},'fresh_helpers':{p.relative_to(base).as_posix():sha(p)for p in(base/'helpers').glob('*.so')},'mapped_libraries':maps,'ldd':subprocess.check_output(['ldd',str(path)],text=True),'scope':'Host-only isolated idle-trim CAPI compile/import. Exact latest3675/3694 GPUobjects/helpers reused with SHAs; no GPUmathchange. Exact complete generated tokens to frozen references; logits bitwise differences diagnostic by human instruction. Full model finite/EOS/repetition/owner/Slurm and throughput remain required.'};out.write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
PY
