if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.3)
    message(FATAL_ERROR "NVIDIA Marlin requires CUDA >=12.3 and C++20")
endif()
set(_w4_root "${PROJECT_SOURCE_DIR}/src/native/cuda/nvidia/ops/w4_marlin")
find_package(Python REQUIRED COMPONENTS Interpreter)
find_program(_w4_patch NAMES patch REQUIRED)
include("${CMAKE_CURRENT_LIST_DIR}/MarlinSource.cmake")
set(_w4_upstream "${vllm_marlin_SOURCE_DIR}")
set(_w4_prepared "${CMAKE_CURRENT_BINARY_DIR}/native-w4-marlin")
set(_w4_inputs
    "${PROJECT_SOURCE_DIR}/scripts/prepare_marlin.py"
    "${PROJECT_SOURCE_DIR}/patches/marlin/w4a16.patch")
foreach(_w4_input IN LISTS _w4_inputs)
    if(NOT EXISTS "${_w4_input}")
        message(FATAL_ERROR
            "Native Marlin is missing ${_w4_input}. Check the pinned source archive or FETCHCONTENT_SOURCE_DIR_VLLM_MARLIN.")
    endif()
endforeach()
foreach(_w4_input IN LISTS _marlin_source_files)
    list(APPEND _w4_inputs "${_w4_upstream}/${_w4_input}")
endforeach()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_w4_inputs})
execute_process(
    COMMAND "${Python_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/scripts/prepare_marlin.py"
        --source-dir "${_w4_upstream}" --output "${_w4_prepared}"
        --patch-executable "${_w4_patch}"
    RESULT_VARIABLE _w4_status ERROR_VARIABLE _w4_prepare_error)
if(NOT _w4_status EQUAL 0)
    message(FATAL_ERROR "Pinned Marlin preparation failed: ${_w4_prepare_error}")
endif()
add_library(infiniops_w4_marlin OBJECT
    "${_w4_root}/repack.cc"
    "${_w4_root}/repack_awq.cu"
    "${_w4_root}/repack_gptq.cu"
    "${_w4_root}/parameters.cu"
    "${_w4_root}/gemm.cu"
    "${_w4_prepared}/marlin.cu"
    "${_w4_prepared}/sm80_kernel_bfloat16_u4_bfloat16.cu"
    "${_w4_prepared}/sm80_kernel_bfloat16_u4b8_bfloat16.cu"
    "${_w4_prepared}/sm80_kernel_float16_u4_float16.cu"
    "${_w4_prepared}/sm80_kernel_float16_u4b8_float16.cu")
set_target_properties(infiniops_w4_marlin PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES 80)
target_include_directories(infiniops_w4_marlin PRIVATE
    "${_w4_prepared}" "${_w4_root}"
    "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/include"
    ${INFINI_RT_INCLUDE_DIRS})
target_link_libraries(infiniops_w4_marlin PRIVATE infinirt CUDA::cudart)
target_compile_definitions(infiniops_w4_marlin PRIVATE
    WITH_NVIDIA=1 INFINI_OPS_MARLIN_KERNEL_ONLY=1)
target_compile_options(infiniops_w4_marlin PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--expt-extended-lambda>)
target_sources(infiniops PRIVATE $<TARGET_OBJECTS:infiniops_w4_marlin>)
