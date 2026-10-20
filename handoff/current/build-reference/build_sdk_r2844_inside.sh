#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export PATH=/root/.local/bin:/opt/dtk/llvm/bin:/opt/dtk/bin:$PATH
export OMP_NUM_THREADS=1
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="/refactor/install/d520-common-r1889/lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
python -u /refactor/incoming/configure_sdk_r2844.py "$LATEST_VARIANT_R2844"
NAME=$(printf '%s' "$LATEST_VARIANT_R2844" | tr '[:upper:]' '[:lower:]')
cmake --build "/refactor/build/ops-${NAME}-3e0d89fe-r2844" -j24
cmake --install "/refactor/build/ops-${NAME}-3e0d89fe-r2844"
export LD_LIBRARY_PATH="/refactor/install/hygon-${NAME}-3e0d89fe-r2844/lib:$LD_LIBRARY_PATH"
python - <<'PY'
import ctypes,hashlib,json,os,pathlib,subprocess
r=pathlib.Path('/refactor');name=os.environ['LATEST_VARIANT_R2844'];p=r/'install'/('hygon-'+name.lower()+'-3e0d89fe-r2844')/'lib/libinfiniops.so';lib=ctypes.CDLL(str(p));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();cfg=r/'results'/f'latest-sdk-r2844-{name}-configure.json';q=json.loads(cfg.read_text());newccl=p.parent/'libinfiniccl.so';assert sha(newccl)==q['CCL_sha256'];out=r/'results'/f'latest-sdk-r2844-{name}-build.json';assert not out.exists();d={'completed':True,'variant':name,'library_sha256':sha(p),'CCL_sha256':sha(newccl),'configure_sha256':sha(cfg),'ldd':subprocess.check_output(['ldd',str(p)],text=True),'pins':q['pins'],'RT_sha256':sha(p.parent/'libinfinirt.so'),'scope':'LatestOpsc86 customHygonSDK build/load only. ActualGPUoperator dtype/math/lifecycle and latestLMmodels/throughput not yet qualified.'};out.write_text(json.dumps(d,indent=2));print('LATEST_SDK_R2844_BUILD_LOAD_OK',json.dumps(d),flush=True)
PY
