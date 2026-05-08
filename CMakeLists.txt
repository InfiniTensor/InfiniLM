cmake_minimum_required(VERSION 3.18)
project(infinilm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Umbrella backend selector. When ON, drives the cuda/*.cu sources through DTK's
# CUDA-compat shim with ENABLE_HYGON_API. The runtime/ccl/core subprojects all
# read `INFINI_NV_BACKEND` to pick which CUDA-API-compatible backend the cuda/
# sources implement (NVIDIA / ILUVATAR / QY / HYGON / ALI).
option(INFINILM_ENABLE_HYGON "Build for Hygon DCU (DTK/HIP) instead of NVIDIA" OFF)
if(INFINILM_ENABLE_HYGON)
    message(STATUS "INFINILM_ENABLE_HYGON=ON → INFINI_NV_BACKEND=HYGON.")
    set(INFINI_NV_BACKEND "HYGON" CACHE STRING "" FORCE)
endif()

option(INFINILM_ENABLE_ASCEND "Build InfiniLM runtime/ccl/core for Ascend CANN/HCCL" OFF)
if(INFINILM_ENABLE_HYGON AND INFINILM_ENABLE_ASCEND)
    message(FATAL_ERROR "INFINILM_ENABLE_HYGON=ON and INFINILM_ENABLE_ASCEND=ON are mutually exclusive. Select exactly one target platform.")
endif()
if(INFINILM_ENABLE_ASCEND)
    set(INFINIRT_ENABLE_NVIDIA OFF CACHE BOOL "" FORCE)
    set(INFINIRT_ENABLE_ASCEND ON CACHE BOOL "" FORCE)
    set(INFINICCL_ENABLE_NVIDIA OFF CACHE BOOL "" FORCE)
    set(INFINICCL_ENABLE_ASCEND ON CACHE BOOL "" FORCE)

    if(NOT DEFINED ASCEND_HOME)
        if(DEFINED ENV{ASCEND_HOME_PATH} AND NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
            set(ASCEND_HOME "$ENV{ASCEND_HOME_PATH}" CACHE PATH "Ascend toolkit root")
        else()
            set(ASCEND_HOME "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "Ascend toolkit root")
        endif()
    endif()
    if(NOT EXISTS "${ASCEND_HOME}")
        message(FATAL_ERROR "INFINILM_ENABLE_ASCEND=ON but ASCEND_HOME was not found. Set ASCEND_HOME_PATH.")
    endif()
    message(STATUS "Using Ascend from `${ASCEND_HOME}`.")
endif()

option(INFINILM_ENABLE_INFINIOPS "Use sibling InfiniOps as the Ascend operator adapter" OFF)
if(INFINILM_ENABLE_ASCEND AND NOT INFINILM_ENABLE_INFINIOPS)
    message(FATAL_ERROR "INFINILM_ENABLE_ASCEND=ON requires INFINILM_ENABLE_INFINIOPS=ON. Specify both options explicitly.")
endif()
if(INFINILM_ENABLE_INFINIOPS AND NOT INFINILM_ENABLE_ASCEND)
    message(FATAL_ERROR "INFINILM_ENABLE_INFINIOPS currently supports only the Ascend adapter. Set INFINILM_ENABLE_ASCEND=ON.")
endif()
set(INFINIOPS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../InfiniOps" CACHE PATH "Path to sibling InfiniOps source tree")
set(INFINIOPS_LIBRARY "" CACHE FILEPATH "Path to libinfiniops.so; auto-detected from Python package when empty")

# Subprojects - each builds as standalone too.
add_subdirectory(runtime)     # libinfinirt.so
add_subdirectory(ccl)         # libinfiniccl.so
add_subdirectory(csrc/core)   # private InfiniLM core

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
    list(FILTER INFINILM_CSRC EXCLUDE REGEX ".*/csrc/core/src/.*")

    pybind11_add_module(_infinilm MODULE ${INFINILM_CSRC})

    target_include_directories(_infinilm PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc/core/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc/core/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/spdlog/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/json/single_include"
    )
    target_link_libraries(_infinilm PRIVATE infinilm_core infinirt infiniccl)
    target_compile_options(_infinilm PRIVATE
        -Wno-deprecated-declarations
        -Wno-unused-function
    )
    # Mirror the API-selector define from runtime/ccl/core onto the pybind module
    # so any csrc/ TU that switches on ENABLE_<X>_API picks the same backend.
    if(INFINILM_ENABLE_ASCEND)
        target_compile_definitions(_infinilm PRIVATE ENABLE_ASCEND_API)
    else()
        target_compile_definitions(_infinilm PRIVATE "ENABLE_${INFINI_NV_BACKEND}_API")
    endif()

    option(INFINILM_ENABLE_KV_CACHING "Enable KV caching op" OFF)
    if(INFINILM_ENABLE_KV_CACHING)
        target_compile_definitions(_infinilm PRIVATE ENABLE_KV_CACHING)
    endif()

    option(INFINILM_USE_CLASSIC_LLAMA "Enable classic LlamaForCausalLM" OFF)
    if(INFINILM_USE_CLASSIC_LLAMA)
        target_compile_definitions(_infinilm PRIVATE USE_CLASSIC_LLAMA)
    endif()

    set(_infinilmlink_rpath "$ORIGIN")
    if(INFINILM_ENABLE_INFINIOPS)
        if(NOT EXISTS "${INFINIOPS_ROOT}/src/operator.h")
            message(FATAL_ERROR "INFINILM_ENABLE_INFINIOPS=ON but ${INFINIOPS_ROOT}/src/operator.h was not found.")
        endif()

        if(NOT INFINIOPS_LIBRARY)
            execute_process(
                COMMAND ${Python3_EXECUTABLE} -c "import pathlib, infini; p=pathlib.Path(infini.__file__).resolve().parent/'libinfiniops.so'; print(p if p.exists() else '')"
                OUTPUT_VARIABLE _infinilmlink_infiniops_lib
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            if(_infinilmlink_infiniops_lib)
                set(INFINIOPS_LIBRARY "${_infinilmlink_infiniops_lib}" CACHE FILEPATH "Path to libinfiniops.so" FORCE)
            endif()
        endif()
        if(NOT INFINIOPS_LIBRARY OR NOT EXISTS "${INFINIOPS_LIBRARY}")
            message(FATAL_ERROR "INFINILM_ENABLE_INFINIOPS=ON but libinfiniops.so was not found. Install InfiniOps first or set INFINIOPS_LIBRARY.")
        endif()

        target_compile_definitions(_infinilm PRIVATE INFINILM_ENABLE_INFINIOPS)
        target_include_directories(_infinilm PRIVATE "${INFINIOPS_ROOT}/src")
        target_link_libraries(_infinilm PRIVATE "${INFINIOPS_LIBRARY}")

        get_filename_component(_infinilmlink_infiniops_lib_dir "${INFINIOPS_LIBRARY}" DIRECTORY)
        string(APPEND _infinilmlink_rpath ";${_infinilmlink_infiniops_lib_dir}")
        if(INFINILM_ENABLE_ASCEND)
            target_include_directories(_infinilm PRIVATE
                "${ASCEND_HOME}/include"
                "${ASCEND_HOME}/include/aclnn"
                "${ASCEND_HOME}/include/aclnnop"
                "${ASCEND_HOME}/include/hccl")
            target_link_libraries(_infinilm PRIVATE
                "${ASCEND_HOME}/lib64/libascendcl.so"
                "${ASCEND_HOME}/lib64/libnnopbase.so"
                "${ASCEND_HOME}/lib64/libopapi.so"
                "${ASCEND_HOME}/lib64/libhccl.so")
            string(APPEND _infinilmlink_rpath ";${ASCEND_HOME}/lib64")

            set(ATB_HOME_DIR "$ENV{ATB_HOME_PATH}")
            if(NOT ATB_HOME_DIR)
                file(GLOB ATB_SEARCH_DIRS "/usr/local/Ascend/nnal/atb/*/atb/cxx_abi_1")
                if(ATB_SEARCH_DIRS)
                    list(SORT ATB_SEARCH_DIRS ORDER DESCENDING)
                    list(GET ATB_SEARCH_DIRS 0 ATB_HOME_DIR)
                endif()
            endif()
            if(ATB_HOME_DIR AND EXISTS "${ATB_HOME_DIR}/include/atb/operation.h")
                target_compile_definitions(_infinilm PRIVATE INFINI_HAS_ATB)
                target_include_directories(_infinilm PRIVATE "${ATB_HOME_DIR}/include")
                target_link_libraries(_infinilm PRIVATE "${ATB_HOME_DIR}/lib/libatb.so")
                string(APPEND _infinilmlink_rpath ";${ATB_HOME_DIR}/lib")
            else()
                message(FATAL_ERROR "INFINILM_ENABLE_ASCEND=ON requires ATB for TopKTopPSampler. Set ATB_HOME_PATH.")
            endif()
        endif()
    endif()

    set(INFINILM_PY_PKG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/python/infinilm" CACHE PATH "infinilm python package dir")
    install(TARGETS _infinilm LIBRARY DESTINATION "${INFINILM_PY_PKG_DIR}/lib")

    set_target_properties(_infinilm PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${INFINILM_PY_PKG_DIR}/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "${_infinilmlink_rpath}"
    )

    # Co-locate shared runtime libs next to _infinilm.so so RPATH=$ORIGIN works
    # without LD_LIBRARY_PATH for both editable installs and wheel installs.
    set_target_properties(infinirt  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${INFINILM_PY_PKG_DIR}/lib")
    set_target_properties(infiniccl PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${INFINILM_PY_PKG_DIR}/lib")
    install(TARGETS infinirt infiniccl LIBRARY DESTINATION "${INFINILM_PY_PKG_DIR}/lib")
    foreach(_fa_tgt flash-attn-nvidia flash-attn-hygon flash_attn_hygon_dlsym)
        if(TARGET ${_fa_tgt})
            set_target_properties(${_fa_tgt} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${INFINILM_PY_PKG_DIR}/lib")
            install(TARGETS ${_fa_tgt} LIBRARY DESTINATION "${INFINILM_PY_PKG_DIR}/lib")
        endif()
    endforeach()
endif()
