#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export PYTHONPYCACHEPREFIX="/local/kimi-embedding-${SLURM_JOB_ID}-${SLURM_STEP_ID}"
BASE=/refactor/candidates/kimi-embedding-registry-r3724
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM
ABI=$(python -c 'import torch;print(int(torch.compiled_with_cxx11_abi()))')
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent/"lib")')"
mkdir "$BASE"
for version in 06d0c03-r3550 5f91ad7-r3672; do
 SDK=/refactor/install/hygon-kimi-$version
 export LD_LIBRARY_PATH="$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
 g++ -std=c++17 -O0 -D_GLIBCXX_USE_CXX11_ABI="$ABI" -I"$SDK/include" -I"$COMMON/third_party/json/include" -I/opt/dtk/cuda/cuda/include /refactor/incoming/kimi_embedding_registry_r3724.cpp -L"$SDK/lib" -linfiniops -linfinirt -o "$BASE/$version"
 "$BASE/$version" > "$BASE/$version.json"
done
python - <<'PY'
import json,pathlib,datetime
r=pathlib.Path('/refactor');b=r/'candidates/kimi-embedding-registry-r3724';d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':True,'comparison':{v:json.loads((b/(v+'.json')).read_text())for v in['06d0c03-r3550','5f91ad7-r3672']},'scope':'Actual embeddingop compile/defaultconfig comparison, noGPU/modelmath.'};configs={v:json.loads((r/'results'/f'latest-sdk-r{v}-Kimi-configure.json').read_text()).get('configuration',{})for v in[3550,3672]};d['configuration_differences']={k:{str(v):configs[v].get(k)for v in[3550,3672]}for k in sorted(set(configs[3550])|set(configs[3672]))if configs[3550].get(k)!=configs[3672].get(k)};(r/'results/kimi-embedding-registry-r3724.json').write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
PY
