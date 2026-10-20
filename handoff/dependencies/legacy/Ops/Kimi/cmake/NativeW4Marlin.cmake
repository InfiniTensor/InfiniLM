if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.3)
    message(FATAL_ERROR "NativeW4Marlin candidate requires CUDA >=12.3 and C++20 source_location")
endif()
set(_w4_root "${PROJECT_SOURCE_DIR}/src/native/cuda/nvidia/ops/w4_marlin")
add_library(infiniops_w4_marlin OBJECT
    "${_w4_root}/repack.cc"
    "${_w4_root}/repack_awq.cu"
    "${_w4_root}/repack_gptq.cu"
    "${_w4_root}/parameters.cu"
    "${_w4_root}/gemm.cu"
    "${_w4_root}/vendor/gemm/nvidia/sm80_kernel_bfloat16_u4_bfloat16.cu"
    "${_w4_root}/vendor/gemm/nvidia/sm80_kernel_bfloat16_u4b8_bfloat16.cu"
    "${_w4_root}/vendor/gemm/nvidia/sm80_kernel_float16_u4_float16.cu"
    "${_w4_root}/vendor/gemm/nvidia/sm80_kernel_float16_u4b8_float16.cu")
set_target_properties(infiniops_w4_marlin PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES 80)
target_include_directories(infiniops_w4_marlin PRIVATE
    "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/include"
    ${INFINI_RT_INCLUDE_DIRS})
target_link_libraries(infiniops_w4_marlin PRIVATE infinirt CUDA::cudart)
target_compile_definitions(infiniops_w4_marlin PRIVATE WITH_NVIDIA=1)
target_compile_options(infiniops_w4_marlin PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--expt-extended-lambda>)
target_sources(infiniops PRIVATE $<TARGET_OBJECTS:infiniops_w4_marlin>)
