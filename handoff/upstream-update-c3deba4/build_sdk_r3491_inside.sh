#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""
export PATH=/root/.local/bin:/opt/dtk/llvm/bin:/opt/dtk/bin:$PATH
export OMP_NUM_THREADS=1
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="/refactor/install/d520-common-r1889/lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
python -u /refactor/incoming/configure_sdk_r3491.py "$LATEST_VARIANT_R3491"
NAME=$(printf '%s' "$LATEST_VARIANT_R3491" | tr '[:upper:]' '[:lower:]')
cmake --build "/refactor/build/ops-${NAME}-c3deba4-r3491" -j24
cmake --install "/refactor/build/ops-${NAME}-c3deba4-r3491"
export LD_LIBRARY_PATH="/refactor/install/hygon-${NAME}-c3deba4-r3491/lib:$LD_LIBRARY_PATH"
python - <<'PY'
import ctypes,hashlib,json,os,pathlib,subprocess
r=pathlib.Path('/refactor');name=os.environ['LATEST_VARIANT_R3491'];p=r/'install'/('hygon-'+name.lower()+'-c3deba4-r3491')/'lib/libinfiniops.so';lib=ctypes.CDLL(str(p));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();cfg=r/'results'/f'latest-sdk-r3491-{name}-configure.json';q=json.loads(cfg.read_text());newccl=p.parent/'libinfiniccl.so';assert sha(newccl)==q['CCL_sha256'];out=r/'results'/f'latest-sdk-r3491-{name}-build.json';assert not out.exists();d={'completed':True,'variant':name,'library_sha256':sha(p),'CCL_sha256':sha(newccl),'configure_sha256':sha(cfg),'ldd':subprocess.check_output(['ldd',str(p)],text=True),'pins':q['pins'],'RT_sha256':sha(p.parent/'libinfinirt.so'),'scope':'LatestOpsbcfe customHygonSDK build/load only. ActualGPUoperator dtype/math/lifecycle and latestLMmodels/throughput not yet qualified.'};out.write_text(json.dumps(d,indent=2));print('LATEST_SDK_R3491_BUILD_LOAD_OK',json.dumps(d),flush=True)
PY
