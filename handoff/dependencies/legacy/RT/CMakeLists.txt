cmake_minimum_required(VERSION 3.18)
project(InfiniRT VERSION 0.1.0 LANGUAGES CXX)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(INFINI_RT_INSTALL_CMAKEDIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/InfiniRT"
    CACHE STRING "Directory for the installed InfiniRT CMake package")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(_DEFAULT_HYGON_DTK_ROOT "/opt/dtk")

function(_infinirt_find_hygon_cuda_root out_var dtk_root)
    set(_candidates
        "${dtk_root}/cuda"
        "${dtk_root}/cuda/cuda")
    file(GLOB _versioned_candidates LIST_DIRECTORIES true "${dtk_root}/cuda/cuda-*")
    list(APPEND _candidates ${_versioned_candidates})
    foreach(_candidate IN LISTS _candidates)
        if(EXISTS "${_candidate}/bin/nvcc")
            set(${out_var} "${_candidate}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${out_var} "" PARENT_SCOPE)
endfunction()

# Options for backends.
option(WITH_CPU "Enable CPU backend" OFF)
option(WITH_NVIDIA "Enable CUDA backend" OFF)
option(WITH_ILUVATAR "Enable Iluvatar GPU backend" OFF)
option(WITH_HYGON "Enable Hygon GPU backend" OFF)
option(WITH_THEAD "Enable T-Head PPU backend" OFF)
option(WITH_METAX "Enable MetaX backend" OFF)
option(WITH_MARS "Enable Mars backend" OFF)
option(WITH_CAMBRICON "Enable Cambricon backend" OFF)
option(WITH_MOORE "Enable Moore backend" OFF)
option(WITH_ASCEND "Enable Ascend backend" OFF)

option(AUTO_DETECT_DEVICES "Automatically detect available devices" OFF)
option(INFINI_RT_BUILD_TESTING "Build InfiniRT tests" OFF)
option(INFINI_RT_BUILD_PERFORMANCE_TESTING "Build InfiniRT performance tests" OFF)
option(INFINI_RT_BUILD_DOCS "Build InfiniRT documentation" OFF)

if(AUTO_DETECT_DEVICES)
    message(STATUS "Auto-detecting available devices...")

    set(WITH_CPU ON)

    file(GLOB NVIDIA_DEV_FILES "/dev/nvidia*")

    if(NVIDIA_DEV_FILES)
        set(WITH_NVIDIA ON)
        message(STATUS "Auto-detected NVIDIA environment.")
    endif()

    file(GLOB ILUVATAR_DEV_FILES "/dev/iluvatar*")

    if(ILUVATAR_DEV_FILES)
        set(WITH_ILUVATAR ON)
        message(STATUS "Auto-detected Iluvatar environment.")
    endif()

    if(DEFINED ENV{DTK_ROOT})
        _infinirt_find_hygon_cuda_root(_HYGON_CUDA_DETECT_ROOT "$ENV{DTK_ROOT}")
        if(_HYGON_CUDA_DETECT_ROOT)
            set(WITH_HYGON ON)
            message(STATUS "Auto-detected Hygon environment from DTK_ROOT.")
        endif()
    elseif(EXISTS "${_DEFAULT_HYGON_DTK_ROOT}")
        _infinirt_find_hygon_cuda_root(_HYGON_CUDA_DETECT_ROOT "${_DEFAULT_HYGON_DTK_ROOT}")
        if(_HYGON_CUDA_DETECT_ROOT)
            set(WITH_HYGON ON)
            message(STATUS "Auto-detected Hygon environment from ${_DEFAULT_HYGON_DTK_ROOT}.")
        endif()
    endif()

    if(DEFINED ENV{HPCC_PATH})
        set(WITH_MARS ON)
        message(STATUS "Auto-detected Mars environment from HPCC_PATH")
    elseif(DEFINED ENV{MACA_PATH})
        set(WITH_METAX ON)
        message(STATUS "Auto-detected MetaX environment from MACA_PATH")
    else()
        execute_process(
            COMMAND sh -c "grep -h 9999 /sys/bus/pci/devices/*/vendor 2>/dev/null"
            OUTPUT_VARIABLE _pci_vendor_output
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        string(FIND "${_pci_vendor_output}" "9999" _found_pos)

        if(_found_pos GREATER -1)
            set(WITH_METAX ON)
            message(STATUS "Detected MetaX GPU from PCI vendor ID 0x9999")
        else()
            set(WITH_METAX OFF)
            message(STATUS "No MetaX GPU detected")
        endif()
    endif()

    if(DEFINED ENV{NEUWARE_HOME})
        set(WITH_CAMBRICON ON)
        message(STATUS "Auto-detected Cambricon environment.")
    endif()

    if(DEFINED ENV{MUSA_ROOT} OR DEFINED ENV{MUSA_HOME} OR DEFINED ENV{MUSA_PATH})
        set(WITH_MOORE ON)
        set(WITH_MOORE ON CACHE BOOL "Enable Moore backend" FORCE)
        message(STATUS "Auto-detected Moore environment.")
    else()
        set(WITH_MOORE OFF)
        set(WITH_MOORE OFF CACHE BOOL "Enable Moore backend" FORCE)
    endif()

    if(DEFINED ENV{ASCEND_HOME_PATH} OR EXISTS "/dev/davinci0")
        set(WITH_ASCEND ON)
        message(STATUS "Auto-detected Ascend environment.")
    endif()

    if(WITH_NVIDIA)
        set(_non_nvidia_gpu_detected FALSE)
        foreach(_gpu_backend WITH_ILUVATAR WITH_HYGON WITH_THEAD WITH_METAX WITH_MARS WITH_CAMBRICON WITH_MOORE WITH_ASCEND)
            if(${_gpu_backend})
                set(_non_nvidia_gpu_detected TRUE)
            endif()
        endforeach()

        if(_non_nvidia_gpu_detected)
            set(WITH_NVIDIA OFF)
            message(STATUS "Skipping auto-detected NVIDIA environment because another GPU backend was detected.")
        endif()
    endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Only one CUDA-like GPU backend can be enabled at a time.
set(_gpu_backend_count 0)
foreach(_gpu_backend WITH_NVIDIA WITH_ILUVATAR WITH_HYGON WITH_THEAD WITH_METAX WITH_MARS WITH_MOORE WITH_CAMBRICON WITH_ASCEND)
    if(${_gpu_backend})
        math(EXPR _gpu_backend_count "${_gpu_backend_count} + 1")
    endif()
endforeach()

if(_gpu_backend_count GREATER 1)
    message(FATAL_ERROR "GPU backend options are mutually exclusive. Build one GPU backend at a time.")
endif()

if(WITH_NVIDIA)
    add_compile_definitions(WITH_NVIDIA=1)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# Iluvatar: CUDA-compatible device, uses `clang++` with `-x ivcore` (not `nvcc`).
if(WITH_ILUVATAR)
    add_compile_definitions(WITH_ILUVATAR=1)
    set(ILUVATAR_ARCH "ivcore20" CACHE STRING "Iluvatar GPU architecture")
    find_program(CLANGXX NAMES clang++)
    if(CLANGXX)
        set(CMAKE_CUDA_COMPILER "${CLANGXX}" CACHE STRING "Iluvatar CUDA compiler (clang++)")
    else()
        set(CMAKE_CUDA_COMPILER "clang++" CACHE STRING "Iluvatar CUDA compiler (clang++)")
    endif()
    set(CMAKE_CUDA_FLAGS "-x ivcore -std=c++17 --cuda-gpu-arch=${ILUVATAR_ARCH} -fPIC -Wno-error=unused-variable -Wno-error=unused-private-field -Wno-unused-variable" CACHE STRING "Iluvatar CUDA flags")
    set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF CACHE BOOL "Disable RDC for Iluvatar")
    message(STATUS "Iluvatar: CUDA compiler ${CMAKE_CUDA_COMPILER}, arch ${ILUVATAR_ARCH}")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

if(WITH_HYGON)
    add_compile_definitions(WITH_HYGON=1)
    if(NOT DEFINED DTK_ROOT)
        set(DTK_ROOT "${_DEFAULT_HYGON_DTK_ROOT}")
    endif()
    if(NOT EXISTS "${DTK_ROOT}")
        message(FATAL_ERROR "`WITH_HYGON` is `ON` but `DTK_ROOT` (`${DTK_ROOT}`) does not exist.")
    endif()

    set(_HYGON_ARCH_DEFAULT "gfx906")
    if(DEFINED ENV{HYGON_ARCH} AND NOT "$ENV{HYGON_ARCH}" STREQUAL "")
        set(_HYGON_ARCH_DEFAULT "$ENV{HYGON_ARCH}")
    else()
        find_program(HYGON_ROCMINFO_EXECUTABLE NAMES rocminfo HINTS "${DTK_ROOT}/bin")
        if(HYGON_ROCMINFO_EXECUTABLE)
            execute_process(
                COMMAND ${HYGON_ROCMINFO_EXECUTABLE}
                OUTPUT_VARIABLE _HYGON_ROCMINFO_OUTPUT
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE)
            string(REGEX MATCH "gfx[0-9]+" _HYGON_ARCH_AUTO "${_HYGON_ROCMINFO_OUTPUT}")
            if(_HYGON_ARCH_AUTO)
                set(_HYGON_ARCH_DEFAULT "${_HYGON_ARCH_AUTO}")
            endif()
        endif()
    endif()

    set(HYGON_ARCH "${_HYGON_ARCH_DEFAULT}" CACHE STRING "Hygon GPU architecture")
    _infinirt_find_hygon_cuda_root(HYGON_CUDA_ROOT "${DTK_ROOT}")

    if(NOT HYGON_CUDA_ROOT)
        message(FATAL_ERROR "`WITH_HYGON` is `ON` but no DTK `nvcc` was found under `${DTK_ROOT}`. Checked `${DTK_ROOT}/cuda/bin/nvcc`, `${DTK_ROOT}/cuda/cuda/bin/nvcc`, and `${DTK_ROOT}/cuda/cuda-*/bin/nvcc`.")
    endif()

    set(CMAKE_CUDA_COMPILER "${HYGON_CUDA_ROOT}/bin/nvcc" CACHE FILEPATH "Hygon CUDA compiler (DTK nvcc)")
    set(CUDAToolkit_ROOT "${HYGON_CUDA_ROOT}" CACHE PATH "Hygon CUDA toolkit root")
    set(CMAKE_CUDA_FLAGS "-std=c++17 -fPIC -arch=${HYGON_ARCH} -Wno-return-type -Wno-error=unused-private-field" CACHE STRING "Hygon CUDA flags")

    set(ENV{CUDA_PATH} "${HYGON_CUDA_ROOT}")
    set(ENV{CUDA_HOME} "${HYGON_CUDA_ROOT}")
    set(ENV{PATH} "${HYGON_CUDA_ROOT}/bin:$ENV{PATH}")

    include_directories("${HYGON_CUDA_ROOT}/include")
    link_directories("${HYGON_CUDA_ROOT}/lib64")
    message(STATUS "Hygon: CUDA compiler ${CMAKE_CUDA_COMPILER}, arch ${HYGON_ARCH}, DTK root ${DTK_ROOT}")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

if(WITH_THEAD)
    add_compile_definitions(WITH_THEAD=1)

    set(THEAD_CUDA_ROOT "")
    foreach(_thead_cuda_env THEAD_CUDA_ROOT PPU_CUDA_ROOT)
        if(NOT THEAD_CUDA_ROOT AND DEFINED ENV{${_thead_cuda_env}} AND
           NOT "$ENV{${_thead_cuda_env}}" STREQUAL "")
            set(THEAD_CUDA_ROOT "$ENV{${_thead_cuda_env}}")
        endif()
    endforeach()
    if(NOT THEAD_CUDA_ROOT AND DEFINED ENV{PPU_SDK_ROOT} AND
       NOT "$ENV{PPU_SDK_ROOT}" STREQUAL "")
        set(THEAD_CUDA_ROOT "$ENV{PPU_SDK_ROOT}/CUDA_SDK")
    endif()
    if(NOT THEAD_CUDA_ROOT AND EXISTS "/usr/local/PPU_SDK/CUDA_SDK")
        set(THEAD_CUDA_ROOT "/usr/local/PPU_SDK/CUDA_SDK")
    endif()

    if(NOT THEAD_CUDA_ROOT OR NOT EXISTS "${THEAD_CUDA_ROOT}/bin/nvcc")
        message(FATAL_ERROR
            "`WITH_THEAD` is `ON` but the Thead PPU CUDA SDK was not found. "
            "Set `THEAD_CUDA_ROOT`, `PPU_CUDA_ROOT`, or `PPU_SDK_ROOT`.")
    endif()

    set(CMAKE_CUDA_COMPILER "${THEAD_CUDA_ROOT}/bin/nvcc"
        CACHE FILEPATH "Thead PPU CUDA compiler" FORCE)
    set(CUDAToolkit_ROOT "${THEAD_CUDA_ROOT}"
        CACHE PATH "Thead PPU CUDA toolkit root" FORCE)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING
            "Thead PPU CUDA architecture" FORCE)
    endif()

    message(STATUS "Thead PPU: CUDA toolkit ${THEAD_CUDA_ROOT}")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

if(WITH_METAX)
    add_compile_definitions(WITH_METAX=1)

    # Normally can be found at: `/opt/maca/`.
    set(MACA_PATH $ENV{MACA_PATH})
    set(CMAKE_C_COMPILER   ${CMAKE_CURRENT_SOURCE_DIR}/scripts/mxcc_wrapper.sh)
    set(CMAKE_CXX_COMPILER ${CMAKE_CURRENT_SOURCE_DIR}/scripts/mxcc_wrapper.sh)

    include_directories("${MACA_PATH}/include")
    link_directories("${MACA_PATH}/lib")

    # Libraries: mcruntime / mcdnn / mcblas.
    find_library(MACA_RUNTIME_LIB NAMES mcruntime HINTS "${MACA_PATH}/lib" REQUIRED)
endif()

if(WITH_MARS)
    add_compile_definitions(WITH_MARS=1)

    # Normally can be found at: `/opt/hpcc/`.
    set(HPCC_PATH $ENV{HPCC_PATH})

    include_directories("${HPCC_PATH}/include")
    link_directories("${HPCC_PATH}/lib")

    find_library(HPCC_RUNTIME_LIB NAMES hcruntime HINTS "${HPCC_PATH}/lib" REQUIRED)
endif()

if(WITH_MOORE)
    add_compile_definitions(WITH_MOORE=1)

    set(MUSA_ROOT "")
    foreach(_musa_env MUSA_ROOT MUSA_HOME MUSA_PATH)
        if(NOT MUSA_ROOT AND DEFINED ENV{${_musa_env}} AND NOT "$ENV{${_musa_env}}" STREQUAL "")
            set(MUSA_ROOT "$ENV{${_musa_env}}")
        endif()
    endforeach()

    if(NOT MUSA_ROOT AND EXISTS "/usr/local/musa")
        set(MUSA_ROOT "/usr/local/musa")
    endif()

    if(NOT MUSA_ROOT)
        message(FATAL_ERROR "`WITH_MOORE` is `ON` but `MUSA_ROOT`/`MUSA_HOME`/`MUSA_PATH` is not set and `/usr/local/musa` was not found.")
    endif()

    if(NOT EXISTS "${MUSA_ROOT}/bin/mcc")
        message(FATAL_ERROR "Could not find `mcc` under `${MUSA_ROOT}/bin`.")
    endif()

    message(STATUS "Using Moore from `${MUSA_ROOT}`.")

    set(CMAKE_C_COMPILER   ${CMAKE_CURRENT_SOURCE_DIR}/scripts/mcc_wrapper.sh)
    set(CMAKE_CXX_COMPILER ${CMAKE_CURRENT_SOURCE_DIR}/scripts/mcc_wrapper.sh)

    include_directories("${MUSA_ROOT}/include")
    link_directories("${MUSA_ROOT}/lib")

    find_library(MUSA_LIB   NAMES musa   HINTS "${MUSA_ROOT}/lib" REQUIRED)
    find_library(MUSART_LIB NAMES musart HINTS "${MUSA_ROOT}/lib" REQUIRED)
endif()

if(WITH_CAMBRICON)
    add_compile_definitions(WITH_CAMBRICON=1)
    set(NEUWARE_HOME $ENV{NEUWARE_HOME})

    include_directories("${NEUWARE_HOME}/include")
    link_directories("${NEUWARE_HOME}/lib")
    link_directories("${NEUWARE_HOME}/lib64")

    # Libraries: `cnrt`.
    find_library(CAMBRICON_RUNTIME_LIB NAMES cnrt HINTS "${NEUWARE_HOME}/lib64" REQUIRED)
endif()

if(WITH_ASCEND)
    add_compile_definitions(WITH_ASCEND=1)
    if(NOT DEFINED ASCEND_HOME)
        if(DEFINED ENV{ASCEND_HOME_PATH} AND NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
            set(ASCEND_HOME "$ENV{ASCEND_HOME_PATH}" CACHE PATH "Ascend toolkit root")
        else()
            set(ASCEND_HOME "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "Ascend toolkit root")
        endif()
    endif()
    if(NOT EXISTS "${ASCEND_HOME}")
        message(FATAL_ERROR "`WITH_ASCEND` is ON but `${ASCEND_HOME}` was not found. Set ASCEND_HOME_PATH.")
    endif()
    message(STATUS "Using Ascend from `${ASCEND_HOME}`.")
endif()

# If all other platforms are not enabled, CPU is enabled by default.
if(NOT WITH_NVIDIA AND NOT WITH_ILUVATAR AND NOT WITH_HYGON AND NOT WITH_THEAD AND NOT WITH_METAX AND NOT WITH_MARS AND NOT WITH_MOORE AND NOT WITH_CAMBRICON AND NOT WITH_ASCEND)
    set(WITH_CPU ON)
    add_compile_definitions(WITH_CPU=1)
endif()

set(INFINI_RT_PUBLIC_HEADER_DEVICES "")
if(WITH_CPU)
    list(APPEND INFINI_RT_PUBLIC_HEADER_DEVICES cpu)
endif()
if(WITH_NVIDIA)
    list(APPEND INFINI_RT_PUBLIC_HEADER_DEVICES nvidia)
endif()
if(WITH_ILUVATAR)
    list(APPEND INFINI_RT_PUBLIC_HEADER_DEVICES iluvatar)
endif()
if(WITH_HYGON)
    list(APPEND INFINI_RT_PUBLIC_HEADER_DEVICES hygon)
endif()
if(WITH_THEAD)
    list(APPEND INFINI_RT_PUBLIC_HEADER_DEVICES thead)
endif()
if(WITH_METAX)
    list(APPEND INFINI_RT_PUBLIC_HEADER_DEVICES metax)
endif()
if(WITH_MARS)
    list(APPEND INFINI_RT_PUBLIC_HEADER_DEVICES mars)
endif()
if(WITH_MOORE)
    list(APPEND INFINI_RT_PUBLIC_HEADER_DEVICES moore)
endif()
if(WITH_CAMBRICON)
    list(APPEND INFINI_RT_PUBLIC_HEADER_DEVICES cambricon)
endif()
if(WITH_ASCEND)
    list(APPEND INFINI_RT_PUBLIC_HEADER_DEVICES ascend)
endif()

find_package(Python COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND ${Python_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_public_headers.py
        --output-dir ${CMAKE_CURRENT_SOURCE_DIR}/generated/include
        --source-output ${CMAKE_CURRENT_SOURCE_DIR}/generated/src/runtime_dispatch.cc
        --devices ${INFINI_RT_PUBLIC_HEADER_DEVICES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE INFINI_RT_PUBLIC_HEADER_RESULT
)
if(NOT INFINI_RT_PUBLIC_HEADER_RESULT EQUAL 0)
    message(FATAL_ERROR "Generating public headers failed")
endif()

add_subdirectory(src)

set(INFINI_RT_PACKAGE_NEEDS_CUDATOOLKIT OFF)
if(WITH_NVIDIA OR WITH_ILUVATAR OR WITH_HYGON OR WITH_THEAD)
    set(INFINI_RT_PACKAGE_NEEDS_CUDATOOLKIT ON)
endif()

set(INFINI_RT_PACKAGE_VENDOR_BACKEND "")
foreach(_backend METAX MARS MOORE CAMBRICON ASCEND)
    if(WITH_${_backend})
        set(INFINI_RT_PACKAGE_VENDOR_BACKEND "${_backend}")
    endif()
endforeach()

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/InfiniRTConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/InfiniRTConfig.cmake"
    INSTALL_DESTINATION "${INFINI_RT_INSTALL_CMAKEDIR}")

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/InfiniRTConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMinorVersion)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/InfiniRTConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/InfiniRTConfigVersion.cmake"
    DESTINATION "${INFINI_RT_INSTALL_CMAKEDIR}")

if(INFINI_RT_BUILD_TESTING OR INFINI_RT_BUILD_PERFORMANCE_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

if(INFINI_RT_BUILD_DOCS)
    include(FetchContent)
    find_package(Doxygen REQUIRED)

    FetchContent_Declare(
        doxygen_awesome
        URL https://github.com/jothepro/doxygen-awesome-css/archive/refs/tags/v2.4.2.tar.gz
        URL_HASH SHA256=6f527e5733723bfd7243c28f24588dca4714cf977fea6147fdb4ae2050244409
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(doxygen_awesome)

    set(INFINI_RT_DOXYGEN_OUTPUT_DIRECTORY
        "${CMAKE_BINARY_DIR}/docs/reference")
    set(INFINI_RT_DOXYGEN_AWESOME_STYLESHEET
        "${doxygen_awesome_SOURCE_DIR}/doxygen-awesome.css")
    set(INFINI_RT_DOXYGEN_STYLESHEET
        "${CMAKE_CURRENT_SOURCE_DIR}/docs/assets/infini-rt.css")
    set(INFINI_RT_DOXYGEN_AWESOME_LICENSE
        "${doxygen_awesome_SOURCE_DIR}/LICENSE")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in"
        "${CMAKE_BINARY_DIR}/docs/Doxyfile"
        @ONLY)

    add_custom_target(infinirt_docs
        COMMAND ${CMAKE_COMMAND} -E make_directory
                "${INFINI_RT_DOXYGEN_OUTPUT_DIRECTORY}"
        COMMAND ${DOXYGEN_EXECUTABLE} "${CMAKE_BINARY_DIR}/docs/Doxyfile"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${INFINI_RT_DOXYGEN_AWESOME_LICENSE}"
                "${INFINI_RT_DOXYGEN_OUTPUT_DIRECTORY}/html/doxygen-awesome.LICENSE"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating InfiniRT documentation"
        VERBATIM)
endif()
