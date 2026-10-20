# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# Copyright (c) 2025 InfiniTensor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Adapted from https://github.com/vllm-project/vllm-ascend/blob/main/csrc/CMakeLists.txt

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(ascend-kernel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RELEASE)
endif()

add_compile_options(-Wunused-value -Wcast-align -Wcast-qual -Wwrite-strings
                    -Wsign-compare -Wextra)

if(${CMAKE_BUILD_TYPE} MATCHES "RELEASE")
    add_compile_options(-O3 -fvisibility=hidden -fvisibility-inlines-hidden
                        -fstack-protector-strong -fPIE -fPIC)
    message(STATUS "Build type set to `RELEASE`.")
else()
    add_compile_options(-g -rdynamic)
endif()

set(PROJECT_OP_SRC_BASE ${PROJECT_SOURCE_DIR})
set(MAIN_SRC_DIR "${PROJECT_OP_SRC_BASE}/../.." CACHE PATH
    "Main InfiniOps source directory.")

include(cmake/config_envs.cmake)
include(cmake/config_ascend.cmake)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "Found `ccache`: ${CCACHE_PROGRAM}.")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

# `CMAKE_LIBRARY_OUTPUT_DIRECTORY` is set by `build.sh` so that the
# standalone `libascend_kernel.so` lands next to `libno_workspace_kernel.a`
# under `<repo>/build/build_ascend_custom/output/`.

# Host-side files.
file(GLOB OP_SRCS
    ${PROJECT_OP_SRC_BASE}/torch_binding.cpp
    ${PROJECT_OP_SRC_BASE}/rms_norm/op_host/rms_norm.cpp
    ${PROJECT_OP_SRC_BASE}/add_rms_norm/op_host/add_rms_norm.cpp
)

# Shared library name — consumed by `kernel_custom.h` variants and by the
# Python side via `import ascend_kernel`.
set(OP_PLUGIN_NAME ascend_kernel)

# Kernel-side files (device code compiled by the `AscendC` toolchain).
ascendc_library(no_workspace_kernel STATIC
    rms_norm/op_kernel/rms_norm.cpp
    add_rms_norm/op_kernel/add_rms_norm.cpp
    reshape_and_cache_flash/op_kernel/reshape_and_cache_flash.cpp
)

# `CANN 8.5.1` installs host objects under `objects-<CONFIG>/`, while
# `recompile_binary.py` only scans the host directory root.
if(TARGET no_workspace_kernel_host)
    add_custom_command(TARGET no_workspace_kernel_host POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -DHOST_DIR=${CMAKE_BINARY_DIR}/no_workspace_kernel_host_dir
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/flatten_host_objects.cmake
        VERBATIM)
endif()

# The kernel translation units include `"data_type_enum.h"` from the main
# project's `src/` so that launcher and device code share one `DataType`
# enum.  `ascendc_library` forwards the interface target's `INCLUDES`
# property to the nested `ExternalProject_Add` (see
# `${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake/legacy_modules/function.cmake`),
# so append the main `src/` dir here.
set_property(TARGET no_workspace_kernel_interface APPEND PROPERTY
    INCLUDES ${MAIN_SRC_DIR})

# Create the shared library `libascend_kernel.so`.
add_library(${OP_PLUGIN_NAME} SHARED ${OP_SRCS})

target_link_libraries(${OP_PLUGIN_NAME} PRIVATE
    no_workspace_kernel
    torch_npu
    ascendcl
    tiling_api
    nnopbase
    opapi
    register
    platform
    ascendalog
    dl
)

target_link_directories(${OP_PLUGIN_NAME} PRIVATE
    ${TORCH_DIR}/lib
    ${TORCH_NPU_DIR}/lib
)

target_include_directories(${OP_PLUGIN_NAME} PRIVATE
    ${PROJECT_OP_SRC_BASE}/utils
    ${PROJECT_SOURCE_DIR}/include
    ${TORCH_DIR}/include
    ${TORCH_DIR}/include/torch/csrc/api/include
    ${TORCH_NPU_DIR}/include/third_party/acl/inc
    ${TORCH_NPU_DIR}/include/third_party/hccl/inc
    ${TORCH_NPU_DIR}/include
    ${PYTHON_INCLUDE_DIR}
    ${ASCEND_INCLUDE_DIR}/external
    ${ASCEND_INCLUDE_DIR}/experiment/platform
    ${ASCEND_INCLUDE_DIR}/experiment/runtime
)
