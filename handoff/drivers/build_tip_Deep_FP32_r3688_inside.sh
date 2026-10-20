#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export PYTHONPYCACHEPREFIX="/local/deep-FP32-build-${SLURM_JOB_ID}-${SLURM_STEP_ID}-${SLURM_PROCID}"
BASE=/refactor/candidates/deep-attention-FP32-5f91ad7-r3688
LM=/refactor/candidates/rebased-5f91ad7-r3668/Model/Deep/InfiniLM
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM
SDK=/refactor/install/hygon-deep-5f91ad7-r3672
HELPER=/refactor/candidates/models-5f91ad7-r3675/Deep/helpers
ABI=$(python -c 'import torch;print(int(torch.compiled_with_cxx11_abi()))')
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="$HELPER:$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$COMMON/python:/old/offline-deps:${PYTHONPATH:-}" OMP_NUM_THREADS=1
unset LD_PRELOAD
test ! -e "$BASE/libmodern_deep_attention_FP32_r3688.so"
source "$BASE/compile.sh"
python - <<'PY'
import ctypes,hashlib,json,datetime,pathlib,os
import torch,infinilm
r=pathlib.Path('/refactor');base=r/'candidates/deep-attention-FP32-5f91ad7-r3688';parent=r/'candidates/models-5f91ad7-r3675/Deep';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();stage=json.loads((r/'results/deep-attention-FP32-r3688-stage.json').read_text());assert stage['ready']
for n,h in stage['files_sha256'].items():assert sha(base/n)==h
for n,h in {**stage['unchanged_latest_GPU_objects'],**stage['unchanged_latest_helpers']}.items():assert sha(parent/n)==h
p=base/'libmodern_deep_attention_FP32_r3688.so';lib=ctypes.CDLL(str(p));lib.modern_deep_precision_policy_r3443.argtypes=[ctypes.c_int];lib.modern_deep_FP32_calls_r3597.argtypes=[ctypes.c_int];lib.modern_deep_FP32_calls_r3597.restype=ctypes.c_uint64
assert lib.modern_deep_precision_policy_r3443(99)==1 and lib.modern_deep_precision_policy_r3443(5)==0 and lib.modern_deep_precision_policy_r3443(0)==0 and all(lib.modern_deep_FP32_calls_r3597(d)==0 for d in range(8))
maps={x:sha(pathlib.Path(x))for x in{v.split()[-1]for v in pathlib.Path('/proc/self/maps').read_text().splitlines()if any(k in v for k in['libinfiniops.so','libinfinirt.so','libinfiniccl.so','libinfinilm_runtime.so'])}};sdk=json.loads((r/'results/latest-sdk-r3672-Deep-build.json').read_text());rt=json.loads((r/'results/common-latest-r3673-Deep-build.json').read_text())
for part,h in [('libinfiniops.so',sdk['library_sha256']),('libinfinirt.so',sdk['RT_sha256']),('libinfiniccl.so',sdk['CCL_sha256']),('libinfinilm_runtime.so',rt['runtime_sha256'])]:assert[v for n,v in maps.items()if part in n]==[h]
out=r/'results/deep-attention-FP32-r3688-build.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':True,'pins':stage['pins'],'library_sha256':sha(p),'source_sha256':stage['files_sha256'],'mapped_libraries':maps,'step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'scope':stage['scope']};out.write_text(json.dumps(d,indent=2));print('LATEST_DEEP_FP32_R3597_BUILD_COMPLETE',d['library_sha256'])
PY
