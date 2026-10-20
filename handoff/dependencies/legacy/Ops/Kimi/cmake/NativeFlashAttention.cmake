# Build the pinned inference kernels separately. No Torch include/link flags or
# Python runtime enter this target; Python is used only to prepare pinned source.
function(infiniops_add_native_flash target)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    find_package(CUDAToolkit REQUIRED)
    if(NOT EXISTS "${FLASH_ATTN_SOURCE_ARCHIVE}")
        message(FATAL_ERROR "Native FlashAttention requires the pinned source archive")
    endif()
    set(_provider "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src/native/cuda/nvidia/ops/flash_attn_with_kvcache")
    set(_varlen_provider "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src/native/cuda/nvidia/ops/flash_attn_varlen_func")
    set(_source "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/..")
    set(_prepared "${CMAKE_CURRENT_BINARY_DIR}/native-flash-inference")
    execute_process(COMMAND "${Python3_EXECUTABLE}"
        "${_source}/scripts/prepare_flash_attention_inference.py"
        --archive "${FLASH_ATTN_SOURCE_ARCHIVE}" --output "${_prepared}"
        RESULT_VARIABLE _status)
    if(NOT _status EQUAL 0)
        message(FATAL_ERROR "Pinned FlashAttention preparation failed")
    endif()
    set(_launches)
    foreach(_flash_kind fp16 bf16)
        if(_flash_kind STREQUAL fp16)
            set(_flash_type "cutlass::half_t")
        else()
            set(_flash_type "cutlass::bfloat16_t")
        endif()
        foreach(_flash_causal false true)
            foreach(_flash_hdim 32 64 96 128 192 256)
                set(_launch "${CMAKE_CURRENT_BINARY_DIR}/flash_${_flash_hdim}_${_flash_kind}_${_flash_causal}.cu")
                configure_file("${_provider}/launch.cu.in" "${_launch}" @ONLY)
                list(APPEND _launches "${_launch}")
                set(_varlen_launch "${CMAKE_CURRENT_BINARY_DIR}/flash_varlen_${_flash_hdim}_${_flash_kind}_${_flash_causal}.cu")
                configure_file("${_varlen_provider}/launch.cu.in" "${_varlen_launch}" @ONLY)
                list(APPEND _launches "${_varlen_launch}")
            endforeach()
        endforeach()
    endforeach()
    add_library(${target} OBJECT "${_provider}/kernel.cc"
        "${_varlen_provider}/kernel.cc" ${_launches})
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 17)
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_definitions(${target} PRIVATE
        FLASH_NAMESPACE=infini_flash FLASHATTENTION_DISABLE_DROPOUT)
    target_include_directories(${target} PRIVATE
        "${_source}/src" "${_source}/include" "${_prepared}"
        "${_prepared}/cutlass" "${_provider}" "${_varlen_provider}"
        ${INFINI_RT_INCLUDE_DIRS})
    target_compile_options(${target} PRIVATE
        # Match the pinned upstream setup.py numerical compilation mode.
        "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--expt-extended-lambda;--use_fast_math>")
    target_link_libraries(${target} PRIVATE CUDA::cudart)
endfunction()
