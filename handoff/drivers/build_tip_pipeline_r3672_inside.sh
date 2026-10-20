#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export PATH=/root/.local/bin:/opt/dtk/llvm/bin:/opt/dtk/bin:$PATH
export OMP_NUM_THREADS=1
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="/refactor/install/stack-5f91ad7-r3669-common/lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
python -u /refactor/incoming/configure_tip_sdk_r3672.py "$MODEL_VARIANT"
NAME=$(printf '%s' "$MODEL_VARIANT" | tr '[:upper:]' '[:lower:]')
cmake --build "/refactor/build/ops-${NAME}-5f91ad7-r3672" -j24
cmake --install "/refactor/build/ops-${NAME}-5f91ad7-r3672"
export LD_LIBRARY_PATH="/refactor/install/hygon-${NAME}-5f91ad7-r3672/lib:$LD_LIBRARY_PATH"
python - <<'PY'
import ctypes,hashlib,json,os,pathlib,subprocess
r=pathlib.Path('/refactor');name=os.environ['MODEL_VARIANT'];p=r/'install'/('hygon-'+name.lower()+'-5f91ad7-r3672')/'lib/libinfiniops.so';lib=ctypes.CDLL(str(p));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();cfg=r/'results'/f'latest-sdk-r3672-{name}-configure.json';q=json.loads(cfg.read_text());newccl=p.parent/'libinfiniccl.so';assert sha(newccl)==q['CCL_sha256'];out=r/'results'/f'latest-sdk-r3672-{name}-build.json';assert not out.exists();d={'completed':True,'variant':name,'library_sha256':sha(p),'CCL_sha256':sha(newccl),'configure_sha256':sha(cfg),'ldd':subprocess.check_output(['ldd',str(p)],text=True),'pins':q['pins'],'RT_sha256':sha(p.parent/'libinfinirt.so'),'scope':'Latest5f91/2b4/1ca with newlyrebuiltcd1RT/51ccCCL, preserved HygonMul/Pad/guard; build/load only, no math/model qualification. ActualGPUoperator dtype/math/lifecycle and latestLMmodels/throughput not yet qualified.'};out.write_text(json.dumps(d,indent=2));print('LATEST_SDK_R3491_BUILD_LOAD_OK',json.dumps(d),flush=True)
PY

source /refactor/incoming/common_tip_r3673_inside.sh
