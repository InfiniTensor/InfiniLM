#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
BASE=/refactor/candidates/kimi-include-order-r3723
LM=/refactor/candidates/kimi-spec-math-5f91ad7-r3683/InfiniLM
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM
SDK=/refactor/install/hygon-kimi-5f91ad7-r3672
export PYTHONPYCACHEPREFIX="/local/kimi-include-${SLURM_JOB_ID}-${SLURM_STEP_ID}"
ABI=$(python -c 'import torch;print(int(torch.compiled_with_cxx11_abi()))')
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent/"lib")')"
export LD_LIBRARY_PATH="$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
mkdir "$BASE"
for variant in math_first registry_first; do
 EXTRA=()
 if [ "$variant" = registry_first ]; then EXTRA=(-DREGISTRY_FIRST_R3717); fi
 g++ -std=c++17 -O0 -D_GLIBCXX_USE_CXX11_ABI="$ABI" "${EXTRA[@]}" -I"$LM/csrc" -I"$SDK/include" -I"$SDK/include/infiniccl" -I"$COMMON/third_party/json/include" -I"$COMMON/third_party/spdlog/include" -I/opt/dtk/cuda/cuda/include /refactor/incoming/kimi_include_order_r3723.cpp -L"$SDK/lib" -linfiniops -linfinirt -o "$BASE/$variant"
 "$BASE/$variant" > "$BASE/$variant.json"
done
python - <<'PY'
import json,pathlib,datetime
r=pathlib.Path('/refactor');b=r/'candidates/kimi-include-order-r3723';d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':True,'comparison':{v:json.loads((b/(v+'.json')).read_text())for v in['math_first','registry_first']},'scope':'Compile-time registry include-order diagnostic, no checkpointload or modelmath.'};(r/'results/kimi-include-order-r3723.json').write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
PY
