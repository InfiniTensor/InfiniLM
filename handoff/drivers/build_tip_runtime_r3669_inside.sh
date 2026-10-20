#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export PYTHONPYCACHEPREFIX="/local/tip-runtime-${SLURM_JOB_ID}-${SLURM_STEP_ID}-${SLURM_PROCID}"
export OMP_NUM_THREADS=1 HYGON_ARCH=gfx936 DTKROOT=/opt/dtk
export LD_LIBRARY_PATH="/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:/opt/dtk/lib:/opt/dtk/rccl/lib:$LD_LIBRARY_PATH"
PREFIX=/refactor/install/stack-5f91ad7-r3669-common
BUILD_RT=/refactor/build/rt-5f91ad7-r3669
BUILD_CCL=/refactor/build/ccl-5f91ad7-r3669
RT_SOURCE=$(python -c 'import json;d=json.load(open("/refactor/results/upstream-5f91ad7-r3667.json"));assert d["completed"] and d["all_recursive_pinned_sources_ready"];print(next(v["remote_path"] for v in d["repositories"] if v["repo"]=="InfiniTensor/InfiniRT").replace("/path/to/cluster-home/hygon-inference-refactor-20261014","/refactor"))')
CCL_SOURCE=$(python -c 'import json;d=json.load(open("/refactor/results/upstream-5f91ad7-r3667.json"));print(next(v["remote_path"] for v in d["repositories"] if v["repo"]=="InfiniTensor/InfiniCCL").replace("/path/to/cluster-home/hygon-inference-refactor-20261014","/refactor"))')
test ! -e "$BUILD_RT/CMakeCache.txt"
test ! -e "$BUILD_CCL/CMakeCache.txt"
test ! -e "$PREFIX/lib/libinfinirt.so"
cmake -S "$RT_SOURCE" -B "$BUILD_RT" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" -DAUTO_DETECT_DEVICES=OFF -DWITH_CPU=ON -DWITH_HYGON=ON -DWITH_NVIDIA=OFF -DHYGON_ARCH=gfx936 -DDTK_ROOT=/opt/dtk -DINFINI_RT_BUILD_TESTING=ON
cmake --build "$BUILD_RT" -j24
ctest --test-dir "$BUILD_RT" --output-on-failure -R 'test_(device_guard|hygon_runtime|runtime_dispatch|core|smoke|hygon_graph)$' -j2
cmake --install "$BUILD_RT"
HIP_CXX=$(command -v hipcc)
cmake -S "$CCL_SOURCE" -B "$BUILD_CCL" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" -DAUTO_DETECT_DEVICES=OFF -DAUTO_DETECT_BACKENDS=OFF -DWITH_HYGON=ON -DWITH_NVIDIA=OFF -DWITH_NCCL=ON -DWITH_OMPI=OFF -DWITH_MPICH=OFF -DBUILD_EXAMPLES=OFF -DCMAKE_CXX_COMPILER="$HIP_CXX" -DCMAKE_CXX_FLAGS="--offload-arch=gfx936"
cmake --build "$BUILD_CCL" -j24
cmake --install "$BUILD_CCL"
export LD_LIBRARY_PATH="$PREFIX/lib:$LD_LIBRARY_PATH"
python - <<'PY'
import pathlib,json,hashlib,datetime,ctypes,subprocess
r=pathlib.Path('/refactor');prefix=r/'install/stack-5f91ad7-r3669-common';source=json.loads((r/'results/upstream-5f91ad7-r3667.json').read_text());pins={v['repo'].split('/')[-1]:v['commit']for v in source['repositories']if v['repo'].startswith('InfiniTensor/')};assert pins['InfiniRT']=='cd1d48258f0a6b8d9f66111d5d6cb1664205e78d'and pins['InfiniCCL']=='51cc9db31d58aac21643d72f0f77cf5c9ecfaee5';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();rt=prefix/'lib/libinfinirt.so';ccl=prefix/'lib/libinfiniccl.so';lib=ctypes.CDLL(str(ccl));assert lib.infinicclGetUniqueId and lib.infinicclCommInitRank and lib.infinicclAllReduce and lib.infinicclReduceScatter and lib.infinicclCommDestroy
d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':True,'pins':pins,'RT_sha256':sha(rt),'CCL_sha256':sha(ccl),'RT_CMakeCache_sha256':sha(r/'build/rt-5f91ad7-r3669/CMakeCache.txt'),'CCL_CMakeCache_sha256':sha(r/'build/ccl-5f91ad7-r3669/CMakeCache.txt'),'RT_tests':(r/'build/rt-5f91ad7-r3669/Testing/Temporary/LastTest.log').read_text(),'CCL_ldd':subprocess.check_output(['ldd',str(ccl)],text=True),'scope':'Fresh latestcd1RT/51ccCCL, installedisolatedprefix; actualHygonRT native/deviceguard/dispatch/graph tests passed. No inherited b413/6f binaries. RealCCL group numeric/lifecycle and wholemodels still required.'};out=r/'results/tip-runtime-r3669-build.json';assert not out.exists();out.write_text(json.dumps(d,indent=2));print('TIP_RUNTIME_R3669_BUILD_COMPLETE',flush=True)
PY
