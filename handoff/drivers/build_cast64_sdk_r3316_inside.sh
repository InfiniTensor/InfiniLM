#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export PATH=/root/.local/bin:/opt/dtk/llvm/bin:/opt/dtk/bin:$PATH
export OMP_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/local/infini-pycache-${SLURM_JOB_ID}-${SLURM_STEP_ID}-${SLURM_PROCID}"
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="/refactor/install/d520-common-r1889/lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
unset LD_PRELOAD
NAME=$(printf '%s' "$CAST64_VARIANT_R3316" | tr '[:upper:]' '[:lower:]')
python -u /refactor/incoming/configure_cast64_sdk_r3316.py "$CAST64_VARIANT_R3316"
cmake --build "/refactor/build/ops-${NAME}-cast64-r3315" -j24
cmake --install "/refactor/build/ops-${NAME}-cast64-r3315"
export LD_LIBRARY_PATH="/refactor/install/hygon-${NAME}-cast64-r3315/lib:$LD_LIBRARY_PATH"
python - <<'PY'
import ctypes,datetime,hashlib,json,os,pathlib,subprocess
r=pathlib.Path('/refactor');name=os.environ['CAST64_VARIANT_R3316'];cfg=json.loads((r/'results'/f'cast64-sdk-r3316-{name}-configure.json').read_text());assert cfg['configured'];p=pathlib.Path(cfg['install']);parent=pathlib.Path(cfg['parent_install']);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();s=json.loads((r/'results/cast64-sdks-r3315-stage.json').read_text());v=s['variants'][name];lib=ctypes.CDLL(str(p/'lib/libinfiniops.so'));assert sha(p/'lib/libinfinirt.so')==v['RT_sha256']and sha(p/'lib/libinfiniccl.so')==v['CCL_sha256'];same={};differences=[]
for q in(parent/'include').rglob('*'):
 if not q.is_file()or q.suffix not in['.h','.hpp','.cuh']:continue
 n=q.relative_to(parent/'include').as_posix()
 if n.startswith('native/'):continue
 target=p/'include'/n
 if not target.exists()or sha(q)!=sha(target):differences.append(n)
 else:same[n]=sha(target)
assert not differences,('publicheaders changed',differences)
maps={q:sha(pathlib.Path(q))for q in{a.split()[-1]for a in pathlib.Path('/proc/self/maps').read_text().splitlines()if any(n in a for n in['libinfiniops.so','libinfinirt.so','libinfiniccl.so'])}}
assert[pathlib.Path(q).resolve()for q in maps if'libinfiniops.so'in q]==[(p/'lib/libinfiniops.so').resolve()]
out=r/'results'/f'cast64-sdk-r3316-{name}-build.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':True,'variant':name,'step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'pins':cfg['pins'],'library_sha256':sha(p/'lib/libinfiniops.so'),'RT_sha256':v['RT_sha256'],'CCL_sha256':v['CCL_sha256'],'public_headers_unchanged_count':len(same),'public_headers_sha256':same,'mapped_libraries':maps,'configure_sha256':sha(r/'results'/f'cast64-sdk-r3316-{name}-configure.json'),'source_stage_sha256':sha(r/'results/cast64-sdks-r3315-stage.json'),'scope':'Freshpermodelc86Ops build/load withoneHygonCastF64patch; allAPIheadersbyteidenticalandRT/CCLhashesunchanged. Onlycompile/load/ABIproof; actualGPU16pairmath/guard/owner/empty/lifecycle andmodels remainunqualified.'};out.write_text(json.dumps(d,indent=2));print('CAST64_SDK_R3316_BUILD_LOAD_COMPLETE',name,d['library_sha256'],flush=True)
PY
