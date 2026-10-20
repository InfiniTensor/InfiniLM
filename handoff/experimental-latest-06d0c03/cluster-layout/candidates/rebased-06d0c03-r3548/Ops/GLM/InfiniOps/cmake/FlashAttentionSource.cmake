# Source inputs only; neither upstream root project is configured or installed.
include(FetchContent)
set(_flash_bundled "${CMAKE_CURRENT_LIST_DIR}/../third_party/flash-attention-source")
if(NOT FETCHCONTENT_SOURCE_DIR_FLASH_ATTENTION AND IS_DIRECTORY "${_flash_bundled}")
    set(FETCHCONTENT_SOURCE_DIR_FLASH_ATTENTION "${_flash_bundled}")
endif()
set(_flash_cutlass_bundled "${CMAKE_CURRENT_LIST_DIR}/../third_party/flash-attention-cutlass-source")
if(NOT FETCHCONTENT_SOURCE_DIR_FLASH_ATTENTION_CUTLASS AND IS_DIRECTORY "${_flash_cutlass_bundled}")
    set(FETCHCONTENT_SOURCE_DIR_FLASH_ATTENTION_CUTLASS "${_flash_cutlass_bundled}")
endif()
cmake_policy(PUSH)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()
FetchContent_Declare(flash_attention
    URL "https://github.com/Dao-AILab/flash-attention/archive/060c9188beec3a8b62b33a3bfa6d5d2d44975fab.tar.gz"
    URL_HASH SHA256=64391cfec73fc5a25932b2c0c2fb3b4d9f97af001164e5220af21049fc17e04e
    SOURCE_SUBDIR csrc/flash_attn/src)
FetchContent_Declare(flash_attention_cutlass
    URL "https://github.com/NVIDIA/cutlass/archive/dc4817921edda44a549197ff3a9dcf5df0636e7b.tar.gz"
    URL_HASH SHA256=f2a3a9df5e6f010c8b02716aa2644a6f071827fafa606fac5f5241cab6a1ab56
    SOURCE_SUBDIR include)
FetchContent_MakeAvailable(flash_attention flash_attention_cutlass)
cmake_policy(POP)

set(_flash_source_files LICENSE AUTHORS csrc/flash_attn/flash_api.cpp)
foreach(_flash_header block_info.h dropout.h flash.h flash_fwd_kernel.h
        flash_fwd_launch_template.h hardware_info.h kernel_traits.h mask.h
        namespace_config.h philox.cuh rotary.h softmax.h static_switch.h utils.h)
    list(APPEND _flash_source_files "csrc/flash_attn/src/${_flash_header}")
endforeach()
foreach(_flash_file IN LISTS _flash_source_files)
    if(NOT EXISTS "${flash_attention_SOURCE_DIR}/${_flash_file}")
        message(FATAL_ERROR "Pinned FlashAttention source is missing ${_flash_file}")
    endif()
endforeach()
foreach(_flash_cutlass_file LICENSE.txt include/cutlass/cutlass.h)
    if(NOT EXISTS "${flash_attention_cutlass_SOURCE_DIR}/${_flash_cutlass_file}")
        message(FATAL_ERROR "Pinned FlashAttention CUTLASS source is missing ${_flash_cutlass_file}")
    endif()
endforeach()
