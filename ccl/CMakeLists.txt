cmake_minimum_required(VERSION 3.18)
project(infiniccl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(INFINICCL_ENABLE_NVIDIA "Enable NVIDIA (NCCL) backend" ON)

# Let caller point to the companion runtime subdirectory, or use an existing target.
set(INFINILM_RUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../runtime" CACHE PATH "Path to runtime subproject")

if(NOT TARGET infinirt)
    # Standalone build: add the runtime subdir.
    add_subdirectory("${INFINILM_RUNTIME_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/_infinirt")
endif()

file(GLOB INFINICCL_CORE_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/infiniccl/*.cc")

set(INFINICCL_CU_SRC "")
set(INFINICCL_DEFS "")
set(INFINICCL_LINK_LIBS "")

set(INFINI_NV_BACKEND "NVIDIA" CACHE STRING "Which CUDA-compatible backend the cuda/ sources implement")

if(INFINICCL_ENABLE_NVIDIA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    file(GLOB _cu_src CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/infiniccl/cuda/*.cu")
    list(APPEND INFINICCL_CU_SRC ${_cu_src})
    list(APPEND INFINICCL_DEFS "ENABLE_${INFINI_NV_BACKEND}_API" ENABLE_CCL)

    find_package(CUDAToolkit REQUIRED)
    list(APPEND INFINICCL_LINK_LIBS CUDA::cudart)

    if(INFINI_NV_BACKEND STREQUAL "HYGON")
        # DTK provides RCCL (NCCL-API-compatible). Auto-detect, fail loud — no fallback.
        find_library(RCCL_LIBRARY NAMES rccl
            HINTS $ENV{INFINI_HYGON_DTK_ROOT}/lib $ENV{ROCM_PATH}/lib
                  $ENV{DTK_ROOT}/lib /opt/dtk/lib)
        find_path(RCCL_INCLUDE_DIR rccl.h
            HINTS $ENV{INFINI_HYGON_DTK_ROOT}/include $ENV{ROCM_PATH}/include
                  $ENV{DTK_ROOT}/include /opt/dtk/include)
        if(NOT RCCL_LIBRARY OR NOT RCCL_INCLUDE_DIR)
            message(FATAL_ERROR
                "RCCL not found. Looked in INFINI_HYGON_DTK_ROOT, ROCM_PATH, /opt/dtk. "
                "Source DTK env (e.g. `source /opt/dtk/env.sh`) or set INFINI_HYGON_DTK_ROOT.")
        endif()
        message(STATUS "RCCL: ${RCCL_LIBRARY}  (include: ${RCCL_INCLUDE_DIR})")
        list(APPEND INFINICCL_LINK_LIBS "${RCCL_LIBRARY}")
        set(NCCL_INCLUDE "${RCCL_INCLUDE_DIR}")
    else()
        # NCCL: use NCCL_ROOT env var if available, else rely on system paths.
        # NCCL_ROOT may also point to a toolkit root with libs under lib64 (Iluvatar Corex).
        set(_nccl_root "$ENV{NCCL_ROOT}")
        if(_nccl_root)
            if(EXISTS "${_nccl_root}/lib/libnccl.so")
                list(APPEND INFINICCL_LINK_LIBS "${_nccl_root}/lib/libnccl.so")
            elseif(EXISTS "${_nccl_root}/lib64/libnccl.so")
                list(APPEND INFINICCL_LINK_LIBS "${_nccl_root}/lib64/libnccl.so")
            else()
                list(APPEND INFINICCL_LINK_LIBS nccl)
            endif()
            set(NCCL_INCLUDE "${_nccl_root}/include")
        else()
            list(APPEND INFINICCL_LINK_LIBS nccl)
            set(NCCL_INCLUDE "")
        endif()
    endif()
endif()

add_library(infiniccl SHARED ${INFINICCL_CORE_SRC} ${INFINICCL_CU_SRC})

target_include_directories(infiniccl
    PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
if(NCCL_INCLUDE)
    target_include_directories(infiniccl PRIVATE "${NCCL_INCLUDE}")
endif()

target_compile_definitions(infiniccl PRIVATE ${INFINICCL_DEFS})

if(INFINICCL_ENABLE_NVIDIA)
    if(INFINI_NV_BACKEND STREQUAL "HYGON")
        set_target_properties(infiniccl PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
            set_target_properties(infiniccl PROPERTIES CUDA_ARCHITECTURES "75")
        endif()
    else()
        set_target_properties(infiniccl PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
            set_target_properties(infiniccl PROPERTIES CUDA_ARCHITECTURES native)
        endif()
    endif()
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
        target_compile_options(infiniccl PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-fPIC>
            $<$<COMPILE_LANGUAGE:CUDA>:-Wno-unknown-cuda-version>
            $<$<COMPILE_LANGUAGE:CUDA>:-Wno-unused-command-line-argument>
        )
    elseif(INFINI_NV_BACKEND STREQUAL "HYGON")
        target_compile_options(infiniccl PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>
        )
    else()
        target_compile_options(infiniccl PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>
        )
    endif()
endif()

target_link_libraries(infiniccl PRIVATE infinirt ${INFINICCL_LINK_LIBS})

install(TARGETS infiniccl LIBRARY DESTINATION lib)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/infiniccl.h DESTINATION include)
