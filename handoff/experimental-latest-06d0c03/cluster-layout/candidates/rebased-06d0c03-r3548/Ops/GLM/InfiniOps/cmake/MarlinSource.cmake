# Source acquisition only; never configure or link the upstream vLLM project.
include(FetchContent)
set(_marlin_bundled "${CMAKE_CURRENT_LIST_DIR}/../third_party/marlin-source")
if(NOT FETCHCONTENT_SOURCE_DIR_VLLM_MARLIN AND IS_DIRECTORY "${_marlin_bundled}")
    # Complete sdists carry the same eleven upstream inputs, without Git metadata.
    set(FETCHCONTENT_SOURCE_DIR_VLLM_MARLIN "${_marlin_bundled}")
endif()
cmake_policy(PUSH)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()
FetchContent_Declare(vllm_marlin
    URL "https://github.com/vllm-project/vllm/archive/ffc4f08c8ee130d4ea6347c1bf31ffd4f8af28ab.tar.gz"
    URL_HASH SHA256=d8f3d663347dff35ea8123d3eb570f67d2dd6c4258ee8b8a8ec2996c42ada678
    SOURCE_SUBDIR csrc/libtorch_stable/quantization/marlin)
FetchContent_MakeAvailable(vllm_marlin)
cmake_policy(POP)

set(_marlin_source_files LICENSE)
foreach(_marlin_file dequant.h kernel.h marlin.cuh marlin_dtypes.cuh
        marlin_mma.h marlin_template.h awq_marlin_repack.cu
        gptq_marlin_repack.cu generate_kernels.py marlin.cu)
    list(APPEND _marlin_source_files
        "csrc/libtorch_stable/quantization/marlin/${_marlin_file}")
endforeach()
list(APPEND _marlin_source_files "csrc/core/scalar_type.hpp")
foreach(_marlin_file IN LISTS _marlin_source_files)
    if(NOT EXISTS "${vllm_marlin_SOURCE_DIR}/${_marlin_file}")
        message(FATAL_ERROR "Pinned Marlin source is missing ${_marlin_file}")
    endif()
endforeach()
