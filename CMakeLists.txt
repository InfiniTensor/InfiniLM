cmake_minimum_required(VERSION 3.18)
project(infinilm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Umbrella backend selector. When ON, drives the cuda/*.cu sources through DTK's
# CUDA-compat shim with ENABLE_HYGON_API. The runtime/ccl/ops subprojects all
# read `INFINI_NV_BACKEND` to pick which CUDA-API-compatible backend the cuda/
# sources implement (NVIDIA / ILUVATAR / QY / HYGON / ALI).
option(INFINILM_ENABLE_HYGON "Build for Hygon DCU (DTK/HIP) instead of NVIDIA" OFF)
if(INFINILM_ENABLE_HYGON)
    message(STATUS "INFINILM_ENABLE_HYGON=ON → INFINI_NV_BACKEND=HYGON.")
    set(INFINI_NV_BACKEND "HYGON" CACHE STRING "" FORCE)
endif()

# Subprojects - each builds as standalone too.
add_subdirectory(runtime)     # libinfinirt.so
add_subdirectory(ccl)         # libinfiniccl.so
add_subdirectory(ops)         # libinfinicore.so + _infinicore python module

# ---- _infinilm pybind module ----
option(INFINILM_BUILD_PYBIND "Build _infinilm python module" ON)

if(INFINILM_BUILD_PYBIND)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _pb_rc
    )
    if(NOT _pb_rc EQUAL 0 OR PYBIND11_CMAKE_DIR STREQUAL "")
        message(FATAL_ERROR "pybind11 not found (pip install pybind11)")
    endif()
    set(pybind11_DIR "${PYBIND11_CMAKE_DIR}")
    find_package(pybind11 CONFIG REQUIRED)

    file(GLOB_RECURSE INFINILM_CSRC CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cc"
    )

    pybind11_add_module(_infinilm MODULE ${INFINILM_CSRC})

    target_include_directories(_infinilm PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/spdlog/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/json/single_include"
    )
    target_link_libraries(_infinilm PRIVATE infinicore infinirt infiniccl)
    target_compile_options(_infinilm PRIVATE
        -Wno-deprecated-declarations
        -Wno-unused-function
    )
    # Mirror the API-selector define from runtime/ccl/ops onto the pybind module
    # so any csrc/ TU that switches on ENABLE_<X>_API picks the same backend.
    target_compile_definitions(_infinilm PRIVATE "ENABLE_${INFINI_NV_BACKEND}_API")

    option(INFINILM_ENABLE_KV_CACHING "Enable KV caching op" OFF)
    if(INFINILM_ENABLE_KV_CACHING)
        target_compile_definitions(_infinilm PRIVATE ENABLE_KV_CACHING)
    endif()

    option(INFINILM_USE_CLASSIC_LLAMA "Enable classic LlamaForCausalLM" OFF)
    if(INFINILM_USE_CLASSIC_LLAMA)
        target_compile_definitions(_infinilm PRIVATE USE_CLASSIC_LLAMA)
    endif()

    set(INFINILM_PY_PKG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/python/infinilm" CACHE PATH "infinilm python package dir")
    install(TARGETS _infinilm LIBRARY DESTINATION "${INFINILM_PY_PKG_DIR}/lib")

    set_target_properties(_infinilm PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${INFINILM_PY_PKG_DIR}/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "$ORIGIN"
    )

    # Co-locate the three core shared libs next to _infinilm.so so RPATH=$ORIGIN works
    # without LD_LIBRARY_PATH for both editable installs and wheel installs.
    set_target_properties(infinirt  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${INFINILM_PY_PKG_DIR}/lib")
    set_target_properties(infiniccl PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${INFINILM_PY_PKG_DIR}/lib")
    set_target_properties(infinicore PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${INFINILM_PY_PKG_DIR}/lib")
    install(TARGETS infinirt infiniccl infinicore LIBRARY DESTINATION "${INFINILM_PY_PKG_DIR}/lib")
    foreach(_fa_tgt flash-attn-nvidia flash-attn-hygon flash_attn_hygon_dlsym)
        if(TARGET ${_fa_tgt})
            set_target_properties(${_fa_tgt} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${INFINILM_PY_PKG_DIR}/lib")
            install(TARGETS ${_fa_tgt} LIBRARY DESTINATION "${INFINILM_PY_PKG_DIR}/lib")
        endif()
    endforeach()
endif()
