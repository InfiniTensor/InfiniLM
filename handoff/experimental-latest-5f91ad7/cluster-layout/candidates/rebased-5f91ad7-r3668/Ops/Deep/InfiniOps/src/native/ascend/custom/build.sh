#!/bin/bash
# Build custom `AscendC` kernels into `libno_workspace_kernel.a` (+ the
# standalone `libascend_kernel.so`).
#
# Intermediate artifacts default to `<repo>/build/build_ascend_custom/`
# so the source tree under `src/` stays free of build output. Override
# via `BUILD_DIR=<abs-path> bash build.sh <soc>` if needed.
set -e

SOC_VERSION="${1:-Ascend910_9382}"

# Detect CANN toolkit path.
_CANN_TOOLKIT_INSTALL_PATH=$(grep "Toolkit_InstallPath" /etc/Ascend/ascend_cann_install.info | awk -F'=' '{print $2}')
source "${_CANN_TOOLKIT_INSTALL_PATH}/set_env.sh"
echo "CANN: ${ASCEND_TOOLKIT_HOME}"

ASCEND_INCLUDE_DIR=${ASCEND_TOOLKIT_HOME}/$(arch)-linux/include

# Resolve build directory.  `<script>/../../..` is `<repo>/`.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REAL_SCRIPT_DIR=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
REPO_ROOT=$(cd "${REAL_SCRIPT_DIR}/../../.." && pwd)
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build/build_ascend_custom}"
OUTPUT_DIR="${BUILD_DIR}/output"
MAIN_SRC_DIR="${MAIN_SRC_DIR:-${REPO_ROOT}/src}"

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}" "${OUTPUT_DIR}"

cmake \
    -DASCEND_HOME_PATH="${ASCEND_HOME_PATH}" \
    -DASCEND_INCLUDE_DIR="${ASCEND_INCLUDE_DIR}" \
    -DSOC_VERSION="${SOC_VERSION}" \
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="${OUTPUT_DIR}" \
    -DMAIN_SRC_DIR="${MAIN_SRC_DIR}" \
    -B "${BUILD_DIR}" \
    -S "${SCRIPT_DIR}"

cmake --build "${BUILD_DIR}" -j 16

echo "Build complete. Output: ${OUTPUT_DIR}"
