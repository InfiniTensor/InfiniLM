#!/bin/bash
# Run inside the pinned DTK container. Builds native code; does not certify GPU correctness.
set -euo pipefail
source /opt/dtk/env.sh
export OMP_NUM_THREADS=1
export PATH=/root/.local/bin:/opt/dtk/llvm/bin:/opt/dtk/bin:$PATH
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
export CUDA_HOME=/opt/dtk/cuda/cuda CUDA_PATH=/opt/dtk/cuda/cuda
export TORCH_LIB_DIR=$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent/"lib")')
export PYBIND11_INCLUDE_DIR=$(python -c 'import pybind11;print(pybind11.get_include())')
ABI=$(python -c 'import torch;print(int(torch.compiled_with_cxx11_abi()))')
export INFINILM_CXX11_ABI=$ABI XMAKE_ROOT=y
export CPLUS_INCLUDE_PATH=$(python -c 'import sysconfig;print(sysconfig.get_path("include"))'):${CPLUS_INCLUDE_PATH:-}
export LD_LIBRARY_PATH="$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export TMPDIR=/local/tmp
python -c 'import sys;assert sys.version_info[:2]==(3,10), "Archived CAPI link names require Python 3.10; port and requalify for another Python ABI"'
test "${HANDOFF_PROFILE:-current}" = current
phase=${1:?Expected runtime, sdk, common, or model}
NAME=${2:-GLM}
case "$NAME" in Kimi|GLM|Deep) ;; *) exit 2;; esac
lower=$(printf '%s' "$NAME" | tr '[:upper:]' '[:lower:]')
jobs=${BUILD_JOBS:-16}
prefix=/refactor/install/hygon-modern
SDK=/refactor/install/hygon-${lower}-3e0d89fe-r2844
LM=/refactor/candidates/rebased-3e0d89fe-r2842/Model/$NAME/InfiniLM
COMMON=/refactor/candidates/rebased-3e0d89fe-r2842/Common/$NAME/InfiniLM
OPS=/refactor/candidates/rebased-3e0d89fe-r2842/Ops/$NAME/InfiniOps
case "$phase" in
runtime)
  cmake -S /refactor/src/RT -B /refactor/build/handoff-rt -DWITH_HYGON=ON -DWITH_NVIDIA=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=$ABI" -DINFINI_RT_BUILD_TESTING=OFF
  cmake --build /refactor/build/handoff-rt -j"$jobs"
  cmake --install /refactor/build/handoff-rt
  cmake -S /refactor/src/CCL -B /refactor/build/handoff-ccl -DWITH_HYGON=ON -DWITH_NCCL=ON -DAUTO_DETECT_DEVICES=OFF -DBUILD_EXAMPLES=OFF -DHYGON_DTK_ROOT=/opt/dtk -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=$ABI"
  cmake --build /refactor/build/handoff-ccl -j"$jobs"
  cmake --install /refactor/build/handoff-ccl
  ;;
sdk)
  test ! -e "$SDK"
  mkdir -p "$SDK"
  cp -a "$prefix/include" "$prefix/lib" "$SDK/"
  export HANDOFF_SDK="$SDK" HANDOFF_OPS="$OPS" HANDOFF_NAME="$NAME" HANDOFF_ABI="$ABI"
  python - <<'PY'
import json,os,pathlib,subprocess
n=os.environ['HANDOFF_NAME'];sdk=os.environ['HANDOFF_SDK'];ops=os.environ['HANDOFF_OPS'];abi=os.environ['HANDOFF_ABI']
cfg=json.loads(pathlib.Path('/handoff-repo/handoff/current/config',n+'-cmake-original.json').read_text())
cfg.update({'CMAKE_INSTALL_PREFIX:PATH':sdk,'INFINI_RT_ROOT:PATH':'/refactor/install/hygon-modern','INFINI_RT_INCLUDE_DIRS:STRING':'/refactor/install/hygon-modern/include','INFINI_RT_LIBRARY:FILEPATH':'/refactor/install/hygon-modern/lib/libinfinirt.so','FETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON:PATH':'/refactor/_upstream_deps/json','CMAKE_CXX_FLAGS:STRING':'-D_GLIBCXX_USE_CXX11_ABI='+abi})
cfg['CMAKE_CUDA_FLAGS:STRING']=cfg['CMAKE_CUDA_FLAGS:STRING'].replace('_GLIBCXX_USE_CXX11_ABI=1','_GLIBCXX_USE_CXX11_ABI='+abi)
subprocess.run(['cmake','-S',ops,'-B','/refactor/build/ops-'+n.lower()+'-3e0d89fe-r2844']+['-D'+k+'='+v for k,v in cfg.items()],check=True)
PY
  cmake --build "/refactor/build/ops-${lower}-3e0d89fe-r2844" -j"$jobs"
  cmake --install "/refactor/build/ops-${lower}-3e0d89fe-r2844"
  ;;
common)
  export INFINI_ROOT="$SDK"
  export LD_LIBRARY_PATH="$SDK/lib:$LD_LIBRARY_PATH"
  cd "$COMMON"
  xmake f -y -m release --cxx11-abi="$ABI" -cv
  xmake build -y -j"$jobs" _infinilm
  xmake install -y infinilm_runtime
  xmake install -y _infinilm
  ;;
model)
  if [ "$NAME" = Kimi ]; then BASE=/refactor/candidates/models-3e0d89fe-r2863/Kimi
  elif [ "$NAME" = GLM ]; then BASE=/refactor/candidates/glm-batch-model-r3072
  else BASE=/refactor/candidates/models-3e0d89fe-r2856/Deep; fi
  HELPER=$BASE/helpers
  export LD_LIBRARY_PATH="$HELPER:$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:$LD_LIBRARY_PATH"
  export PYTHONPATH="$COMMON/python:/old/offline-deps:${PYTHONPATH:-}"
  # Use the archived fresh GPU compiler section, excluding historical receipt admission.
  export HANDOFF_HELPER_SCRIPT=/handoff-repo/handoff/current/build-reference/build_models_latest_r2856_inside.sh
  python - <<'PY' > /local/tmp/build-helpers.sh
import os,pathlib
s=pathlib.Path(os.environ['HANDOFF_HELPER_SCRIPT']).read_text();s=s[s.index('NVCC='):s.index('source "$BASE/compile.sh"')]
print(s)
PY
  export LATEST_VARIANT_R2856=$NAME
  source /local/tmp/build-helpers.sh
  source "$BASE/compile.sh"
  ;;
*) printf 'Unknown build phase: %s\n' "$phase" >&2; exit 2;;
esac
printf 'Build phase finished: %s %s. GPU and model audit still required.\n' "$phase" "$NAME"
