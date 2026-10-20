#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export PATH=/root/.local/bin:/opt/dtk/llvm/bin:/opt/dtk/bin:$PATH
NAME=$(printf '%s' "$MODEL_VARIANT" | tr '[:upper:]' '[:lower:]')
LM=/refactor/candidates/rebased-06d0c03-r3548/Common/$MODEL_VARIANT/InfiniLM
SDK=/refactor/install/hygon-${NAME}-06d0c03-r3550
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')"
export INFINI_ROOT="$SDK" CUDA_HOME=/opt/dtk/cuda/cuda CUDA_PATH=/opt/dtk/cuda/cuda
export INFINILM_CXX11_ABI="$(python -c 'import torch;print(int(torch.compiled_with_cxx11_abi()))')"
export LD_LIBRARY_PATH="$SDK/lib:$TORCH_LIB_DIR:$CUDA_HOME/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export PYBIND11_INCLUDE_DIR=/refactor/build-deps/pybind11-3.0.4/include
PYTHON_INCLUDE_DIR=$(python -c 'import sysconfig;print(sysconfig.get_path("include"))')
export CPLUS_INCLUDE_PATH="$PYTHON_INCLUDE_DIR:${CPLUS_INCLUDE_PATH:-}"
export XMAKE_PROGRAMDIR=/root/.local/share/xmake XMAKE_GLOBALDIR=/root XMAKE_ROOT=y OMP_NUM_THREADS=1
mkdir /local/compiler-tmp
export TMPDIR=/local/compiler-tmp
unset INFINILM_DENSE_PREFILL_F32_R2475 INFINILM_MXFP4_TILE_TP_R2559 LD_PRELOAD
python - <<'PY'
import hashlib,json,os,pathlib
r=pathlib.Path('/refactor');name=os.environ['MODEL_VARIANT'];base=r/'candidates/rebased-06d0c03-r3548/Common'/name;root=base/'InfiniLM';m=json.loads((base/'migration-manifest.json').read_text());assert m['ready']and m['pins']['InfiniLM']=='06d0c03cc860d730afe5f35e41e53a443221da23'
for v in m['changes']:
 if'merged_sha256'in v:assert hashlib.sha256((root/v['path']).read_bytes()).hexdigest()==v['merged_sha256']
cpu=json.loads((r/'results/cpu-copy-r2515-stage.json').read_text())['variants'][name];assert hashlib.sha256((root/'csrc/runtime/tensor/copy.cc').read_bytes()).hexdigest()==cpu['copy_source_sha256'];assert hashlib.sha256((root/'csrc/runtime/tensor/cpu_strided_copy_r2515.hpp').read_bytes()).hexdigest()==cpu['header_sha256']
sdk=json.loads((r/'results'/f'latest-sdk-r3550-{name}-build.json').read_text());assert sdk['completed']and sdk['pins']==m['pins'];assert hashlib.sha256((r/'install'/f'hygon-{name.lower()}-06d0c03-r3550/lib/libinfiniops.so').read_bytes()).hexdigest()==sdk['library_sha256']
assert not list((root/'python').rglob('*.so'))
PY
cd "$LM"
xmake f -y -m release --cxx11-abi="$INFINILM_CXX11_ABI" -cv
xmake build -y -j16 _infinilm
xmake install -y infinilm_runtime
xmake install -y _infinilm
export LD_LIBRARY_PATH="$SDK/lib:$LM/python:$LM/python/infinilm/_lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$LM/python:/old/offline-deps:${PYTHONPATH:-}"
mkdir "$LM/native-checks-r3551"
g++ -std=c++17 -O2 -pthread -I"$LM/csrc/engine" "$LM/csrc/engine/rank_barrier.cpp" "$LM/test/native/rank_barrier_test.cpp" -o "$LM/native-checks-r3551/rank_barrier_test"
timeout 20 "$LM/native-checks-r3551/rank_barrier_test"
g++ -std=c++17 -O2 -I"$LM/csrc/runtime/tensor" /refactor/incoming/cpu_strided_copy_r2515_test.cpp -o "$LM/native-checks-r3551/cpu_copy_test"
timeout 60 "$LM/native-checks-r3551/cpu_copy_test" > "$LM/native-checks-r3551/cpu_copy_test.json"
timeout 60 python "$LM/test/runtime/test_quant_config_guard.py"
python - <<'PY'
import ctypes,datetime,hashlib,json,os,pathlib
import torch,infinilm
from infinilm.lib import _infinilm
r=pathlib.Path('/refactor');name=os.environ['MODEL_VARIANT'];root=r/'candidates/rebased-06d0c03-r3548/Common'/name/'InfiniLM';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();native=pathlib.Path(_infinilm.__file__).resolve();assert native.is_relative_to(root/'python')and pathlib.Path(infinilm.__file__).resolve().is_relative_to(root/'python');rt=root/'python/infinilm/_lib/libinfinilm_runtime.so';lib=ctypes.CDLL(str(rt));lib.modern_dense_prefill_stats_r2475.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];lib.modern_cpu_copy_stats_r2515.argtypes=[ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
values=(ctypes.c_uint64*5)();assert lib.modern_dense_prefill_stats_r2475(0,values,5)==0 and list(values)==[0]*5;assert lib.modern_cpu_copy_stats_r2515(values,5)==0 and list(values)==[0]*5
maps={p:sha(pathlib.Path(p))for p in{v.split()[-1]for v in pathlib.Path('/proc/self/maps').read_text().splitlines()if any(k in v for k in['libinfiniops.so','libinfiniccl.so','libinfinirt.so','libinfinilm_runtime.so'])}};sdk=json.loads((r/'results'/f'latest-sdk-r3550-{name}-build.json').read_text());assert[h for p,h in maps.items()if'libinfiniops.so'in p]==[sdk['library_sha256']];assert[p for p in maps if'libinfinilm_runtime.so'in p]==[str(rt)];assert[h for p,h in maps.items()if'libinfiniccl.so'in p]==[sdk['CCL_sha256']];assert[h for p,h in maps.items()if'libinfinirt.so'in p]==[sdk['RT_sha256']]
test=json.loads((root/'native-checks-r3551/cpu_copy_test.json').read_text());assert test['completed']and test['full_storage_byte_cases']==9606;out=r/'results'/f'common-latest-r3551-{name}-build.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':True,'variant':name,'pins':sdk['pins'],'runtime_sha256':sha(rt),'native_sha256':sha(native),'mapped_libraries':maps,'CPU_copy_test':test,'quant_guard_passed':True,'rank_barrier_passed':True,'source_stage_sha256':sha(r/'results/rebased-06d0c03-r3548-stage.json'),'scope':'Fresh06d/046/f998 Runtime+Python build/import and original9606CPUbyte tests. Defaultoff prefill/tiled numerical candidates. ActualGPU/native ownership and rebuilt model/helper QA+performance still required; no old throughput inherited.'};out.write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
PY
