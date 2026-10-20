#!/bin/bash
set -euo pipefail
python -u /refactor/incoming/stage_kimi_token_only_r963.py
LM=/refactor/candidates/kimi-token-only-r963/InfiniLM
SOURCE=/refactor/candidates/kimi-trace-r3/InfiniLM
SDK=/refactor/install/hygon-kimi-static-r330
BASE=/refactor/candidates/kimi-token-only-r963
HELPER=/refactor/candidates/kimi-grouped-fused-fp32-r21/lib
export LD_LIBRARY_PATH="$HELPER:$SDK/lib:$SOURCE/python:$SOURCE/python/infinilm/_lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=1
ABI=$(python -c 'import torch;print(int(torch.compiled_with_cxx11_abi()))')
test ! -e "$BASE/libmodern_kimi_token_only_r963.so"
g++ -std=c++17 -O2 -shared -fPIC -pthread -D_GLIBCXX_USE_CXX11_ABI="$ABI" -DINFINICORE_RANDOM_SAMPLE_OMITTED \
 "$BASE/kimi_engine_probe_r963.cpp" "$LM/csrc/engine/infer_engine.cpp" "$LM/csrc/engine/rank_worker.cpp" "$LM/csrc/engine/request_sampling.cpp" \
 "$LM/csrc/config/config_factory.cpp" "$LM/csrc/config/model_config.cpp" "$LM/csrc/config/quant_config.cpp" "$LM/csrc/models/models_registry.cpp" "$LM/csrc/models/model_factory.cpp" "$LM"/csrc/models/kimi_k3/*.cpp \
 -I"$LM/csrc" -I"$SDK/include" -I"$SDK/include/infiniccl" -I"$SOURCE/third_party/json/include" -I"$SOURCE/third_party/spdlog/include" -I/opt/dtk/cuda/cuda/include \
 -L"$HELPER" -lkimi_grouped_prefill -L"$SOURCE/python" -l:_infinilm.cpython-310-x86_64-linux-gnu.so -L"$SOURCE/python/infinilm/_lib" -linfinilm_runtime \
 -L"$SDK/lib" -linfiniops -linfiniccl -linfinirt -L/opt/dtk/cuda/cuda/lib64 -lcudart -Wl,-Bsymbolic -Wl,--no-undefined -o "$BASE/libmodern_kimi_token_only_r963.so"
python - <<'PY'
import ctypes,hashlib,json,pathlib
r=pathlib.Path('/refactor');base=r/'candidates/kimi-token-only-r963';library=base/'libmodern_kimi_token_only_r963.so';lib=ctypes.CDLL(str(library));assert lib.modern_kimi_engine_create_r963 and lib.modern_kimi_engine_output_metrics_r963
unit=json.loads((r/'results/kimi-minpages-contract-r914.json').read_text());assert unit['completed']and unit['PP_all93_and_neighbor_prefix_passed'];parent=json.loads((r/'results/kimi-minpages-r914-build.json').read_text());assert parent['cpu_contract_passed']and parent['cache_profiles_passed']
paths=sorted({v.split()[-1]for v in pathlib.Path('/proc/self/maps').read_text().splitlines()if'libinfiniops.so'in v});assert len(paths)==1;assert hashlib.sha256(pathlib.Path(paths[0]).read_bytes()).hexdigest()==parent['sdk_sha256']=='6210897fc1f077bc5fc2e10afc9b8b7141fc4c6c608be3f4cb2d896219560104'
p=r/'results/kimi-token-only-r963-build.json';assert not p.exists();p.write_text(json.dumps({'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),'sdk_sha256':parent['sdk_sha256'],'helper_sha256':parent['helper_sha256'],'source_manifest_sha256':hashlib.sha256((base/'InfiniLM/source-manifest-r963.json').read_bytes()).hexdigest(),'PP_boundaries':[0,24,47,71,93],'parent914_contract_sha256':hashlib.sha256((r/'results/kimi-minpages-contract-r914.json').read_bytes()).hexdigest(),'scope':'963modelcsrc unchanged914, onlyprivateCAPI opt-in omitCPUlogits pluswallmetrics. Defaultfulloutputcompatible. ParentCPUcontract verified; fullmodelA/B andoutputcheckpoints required.'},indent=2));print('KIMI_TOKEN_ONLY_R963_BUILD_OK')
PY
