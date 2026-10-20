cmake_minimum_required(VERSION 3.18)
project(InfiniCCL VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =========================================================
# --- DEVICE OPTIONS (Hardware Runtimes) ---
# =========================================================
option(WITH_NVIDIA      "Enable NVIDIA GPU support"     OFF)
option(WITH_ILUVATAR    "Enable ILUVATAR GPU support"   OFF)
option(WITH_METAX       "Enable MetaX GPU support"      OFF)
option(WITH_MOORE       "Enable Moore GPU support"      OFF)
option(WITH_CAMBRICON   "Enable Cambricon MLU support"  OFF)
option(WITH_HYGON       "Enable Hygon DCU support"      OFF)
option(WITH_THEAD       "Enable T-Head PPU support"     OFF)

set(WITH_CPU ON CACHE INTERNAL "CPU backend is always enabled")

# =========================================================
# --- BACKEND OPTIONS (Communication Protocols) ---
# =========================================================
option(WITH_OMPI "Enable OpenMPI backend" OFF)
option(WITH_MPICH "Enable MPICH backend" OFF)
option(WITH_NCCL "Enable NCCL backend"    OFF)
option(WITH_MCCL "Enable MCCL backend"    OFF)
option(WITH_CNCL "Enable CNCL backend"    OFF)

# =========================================================
# --- MISC. BUILD OPTIONS ---
# =========================================================
option(AUTO_DETECT_DEVICES "Automatically detect available devices" ON)
option(AUTO_DETECT_BACKENDS "Automatically detect available backends" OFF)
option(BUILD_EXAMPLES "Build internal examples" ON)

function(_collect_thead_hints out_var)
    if(NOT DEFINED _THEAD_HINTS_MEMO)
        set(_hints
            /usr/local/PPU_SDK
            /usr/local/PPU_SDK/CUDA_SDK
        )

        foreach(_env IN ITEMS PPU_SDK PPU_HOME PPU_PATH)
            if(DEFINED ENV{${_env}} AND NOT "$ENV{${_env}}" STREQUAL "")
                list(APPEND _hints
                    "$ENV{${_env}}"
                    "$ENV{${_env}}/CUDA_SDK"
                )
            endif()
        endforeach()

        foreach(_env IN ITEMS CUDA_SDK CUDA_HOME CUDA_PATH CUDA_TOOLKIT_ROOT)
            if(DEFINED ENV{${_env}} AND NOT "$ENV{${_env}}" STREQUAL "")
                list(APPEND _hints
                    "$ENV{${_env}}"
                    "$ENV{${_env}}/.."
                )
            endif()
        endforeach()

        list(REMOVE_DUPLICATES _hints)
        set(_THEAD_HINTS_MEMO "${_hints}")
        set(_THEAD_HINTS_MEMO "${_hints}" PARENT_SCOPE)
    endif()

    set(${out_var} "${_THEAD_HINTS_MEMO}" PARENT_SCOPE)
endfunction()

# =========================================================
# --- AUTO-DETECTION: DEVICES ---
# =========================================================
if(AUTO_DETECT_DEVICES)
    message(STATUS "Auto-detecting available devices...")

    # NVIDIA
    set(NVIDIA_REQUESTED ${WITH_NVIDIA})
    set(NVIDIA_FOUND FALSE)

    file(GLOB NVIDIA_DEV_FILES "/dev/nvidia0")

    if(NVIDIA_DEV_FILES)
        set(NVIDIA_FOUND TRUE)
    elseif(NOT NVIDIA_FOUND)
        find_program(NVIDIA_SMI_PATH nvidia-smi)
        if(NVIDIA_SMI_PATH)
            execute_process(
                COMMAND ${NVIDIA_SMI_PATH} -L
                RESULT_VARIABLE SMI_RESULT
                OUTPUT_QUIET
                ERROR_QUIET
            )
            if(SMI_RESULT EQUAL 0)
                set(NVIDIA_FOUND TRUE)
            endif()
        endif()
    endif()

    # Iluvatar
    set(ILUVATAR_FOUND FALSE)

    file(GLOB ILUVATAR_DEV_FILES "/dev/iluvatar*")

    if(ILUVATAR_DEV_FILES)
        set(ILUVATAR_FOUND TRUE)
    elseif(NOT ILUVATAR_FOUND)
        find_program(ILUVATAR_SMI_PATH ixsmi)
        if(ILUVATAR_SMI_PATH)
            execute_process(
                COMMAND ${ILUVATAR_SMI_PATH} -L
                RESULT_VARIABLE SMI_RESULT
                OUTPUT_QUIET
                ERROR_QUIET
            )
            if(SMI_RESULT EQUAL 0)
                set(ILUVATAR_FOUND TRUE)
            endif()
        endif()
    endif()

    if(ILUVATAR_FOUND)
        set(WITH_ILUVATAR ON)
        message(STATUS "Auto-detected Iluvatar environment.")
    else()
        message(STATUS "Iluvatar environment not detected.")
    endif()

    # MetaX
    if(DEFINED ENV{MACA_PATH})
        set(WITH_METAX ON)
        message(STATUS "Auto-detected MetaX environment from `MACA_PATH`")
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

    # MThreads (Moore)
    set(MOORE_FOUND FALSE)

    file(GLOB MOORE_DEV_FILES "/dev/mtgpu*")

    if(DEFINED ENV{MUSA_ROOT} OR DEFINED ENV{MUSA_HOME} OR DEFINED ENV{MUSA_PATH})
        set(MOORE_FOUND TRUE)
    elseif(MOORE_DEV_FILES)
        set(MOORE_FOUND TRUE)
    elseif(NOT MOORE_FOUND)
        find_program(MOORE_SMI_PATH mthreads-gmi)

        if(MOORE_SMI_PATH)
            execute_process(
                COMMAND ${MOORE_SMI_PATH} -L
                RESULT_VARIABLE SMI_RESULT
                OUTPUT_QUIET
                ERROR_QUIET
            )
            if(SMI_RESULT EQUAL 0)
                set(MOORE_FOUND TRUE)
            endif()
        endif()
    endif()

    if(MOORE_FOUND)
        set(WITH_MOORE ON)
        message(STATUS "Auto-detected Moore environment.")
    else()
        message(STATUS "Moore environment not detected.")
    endif()

    # Cambricon
    set(CAMBRICON_FOUND FALSE)

    if(DEFINED ENV{NEUWARE_HOME})
        set(CAMBRICON_FOUND TRUE)
        message(STATUS "Auto-detected Cambricon environment from `NEUWARE_HOME`")
    else()
        find_program(CNMON_PATH cnmon)
        if(CNMON_PATH)
            execute_process(
                COMMAND ${CNMON_PATH} -l
                RESULT_VARIABLE CNMON_RESULT
                OUTPUT_QUIET
                ERROR_QUIET
            )
            if(CNMON_RESULT EQUAL 0)
                set(CAMBRICON_FOUND TRUE)
            endif()
        endif()
    endif()

    if(CAMBRICON_FOUND)
        set(WITH_CAMBRICON ON)
        message(STATUS "Cambricon environment detected.")
    else()
        set(WITH_CAMBRICON OFF)
        message(STATUS "Cambricon environment not detected.")
    endif()

    # Hygon DCU
    if(NOT WITH_HYGON)
        set(HYGON_FOUND FALSE)

        if((DEFINED ENV{DTKROOT} OR DEFINED ENV{DTK_ROOT} OR DEFINED ENV{ROCM_PATH})
           AND EXISTS "/dev/kfd")
            set(HYGON_FOUND TRUE)
        elseif(EXISTS "/opt/dtk" AND EXISTS "/dev/kfd")
            set(HYGON_FOUND TRUE)
        else()
            find_program(HYGON_SMI_PATH
                NAMES hy-smi rocm-smi
                HINTS /opt/hyhal/bin /opt/dtk/bin
            )
            if(HYGON_SMI_PATH)
                execute_process(
                    COMMAND ${HYGON_SMI_PATH}
                    RESULT_VARIABLE SMI_RESULT
                    OUTPUT_QUIET
                    ERROR_QUIET
                )
                if(SMI_RESULT EQUAL 0)
                    set(HYGON_FOUND TRUE)
                endif()
            endif()
        endif()

        if(HYGON_FOUND)
            set(WITH_HYGON ON)
        endif()
    endif()

    if(WITH_HYGON)
        message(STATUS "Hygon environment detected.")
    else()
        message(STATUS "Hygon environment not detected.")
    endif()

    # T-Head
    set(THEAD_FOUND FALSE)

    if(WITH_THEAD)
        set(THEAD_FOUND TRUE)
    else()
        _collect_thead_hints(_THEAD_HINTS)
        foreach(_thead_hint IN LISTS _THEAD_HINTS)
            if(EXISTS "${_thead_hint}/CUDA_SDK" OR EXISTS "${_thead_hint}/bin/ppu-smi" OR EXISTS "${_thead_hint}/ppu-smi/bin/ppu-smi")
                set(THEAD_FOUND TRUE)
            endif()
        endforeach()

        if(NOT THEAD_FOUND)
            find_program(THEAD_SMI_PATH
                NAMES ppu-smi nvidia-smi
                HINTS /usr/local/PPU_SDK/bin /usr/local/PPU_SDK/ppu-smi/bin
            )
            if(THEAD_SMI_PATH)
                execute_process(
                    COMMAND ${THEAD_SMI_PATH} -L
                    RESULT_VARIABLE SMI_RESULT
                    OUTPUT_VARIABLE SMI_OUTPUT
                    ERROR_QUIET
                )
                if(SMI_RESULT EQUAL 0 AND SMI_OUTPUT MATCHES "PPU")
                    set(THEAD_FOUND TRUE)
                endif()
            endif()
        endif()
    endif()

    if(THEAD_FOUND)
        set(WITH_THEAD ON)
        message(STATUS "Auto-detected T-Head environment.")
    else()
        message(STATUS "T-Head environment not detected.")
    endif()

    if(THEAD_FOUND AND NVIDIA_FOUND AND NOT NVIDIA_REQUESTED)
        set(NVIDIA_FOUND FALSE)
        set(WITH_NVIDIA OFF CACHE BOOL "Enable NVIDIA GPU support" FORCE)
        message(STATUS "T-Head environment detected; ignoring NVIDIA auto-detection.")
    endif()

    if(NVIDIA_FOUND)
        set(WITH_NVIDIA ON)
        message(STATUS "Auto-detected NVIDIA environment.")
    else()
        message(STATUS "NVIDIA environment not detected.")
    endif()

endif()

# =========================================================
# --- AUTO-DETECTION: BACKENDS ---
# =========================================================
if(AUTO_DETECT_BACKENDS)
    message(STATUS "Auto-detecting available communication backends...")

    # Detect MPI Dependencies
    find_program(MPIRUN_PATH NAMES mpirun mpiexec)
    find_program(MPICC_PATH NAMES mpicc)
    
    if(MPIRUN_PATH AND MPICC_PATH)
        # Query the active `mpirun` variant.
        execute_process(
            COMMAND ${MPIRUN_PATH} --version
            OUTPUT_VARIABLE MPI_VERSION_OUT
            ERROR_VARIABLE MPI_VERSION_ERR
        )
        set(MPI_VERSION_INFO "${MPI_VERSION_OUT}${MPI_VERSION_ERR}")

        if(MPI_VERSION_INFO MATCHES "Open MPI" OR MPI_VERSION_INFO MATCHES "PRRTE")
            set(WITH_OMPI ON)
            message(STATUS "Auto-detected OpenMPI environment.")
        elseif(MPI_VERSION_INFO MATCHES "MPICH" OR MPI_VERSION_INFO MATCHES "Hydra" OR MPI_VERSION_INFO MATCHES "Intel\\(R\\) MPI")
            set(WITH_MPICH ON)
            message(STATUS "Auto-detected MPICH/Hydra environment.")
        else()
            # Fallback to OMPI if it's an unrecognized wrapper.
            set(WITH_OMPI ON)
            message(STATUS "Unknown MPI implementation. Defaulting to `WITH_OMPI=ON`")
        endif()
    endif()

    # Detect NCCL Dependencies
    if(WITH_NVIDIA OR WITH_THEAD OR WITH_ILUVATAR OR WITH_HYGON)
        set(_NCCL_HINTS)
        set(_NCCL_HEADER_NAMES nccl.h)
        set(_NCCL_LIBRARY_NAMES nccl)
        set(_NCCL_INCLUDE_SUFFIXES include)
        set(_NCCL_LIBRARY_SUFFIXES lib lib64)

        if(WITH_THEAD)
            _collect_thead_hints(_THEAD_HINTS)
            set(_NCCL_HINTS ${_THEAD_HINTS})
            list(APPEND _NCCL_INCLUDE_SUFFIXES targets/x86_64-linux/include)
            list(APPEND _NCCL_LIBRARY_SUFFIXES targets/x86_64-linux/lib)
            list(REMOVE_DUPLICATES _NCCL_HINTS)
            message(STATUS "T-Head detected. Searching for NCCL in ${_NCCL_HINTS}")
        elseif(WITH_ILUVATAR)
            set(_NCCL_HINTS /usr/local/corex)
            message(STATUS "Iluvatar detected. Searching for NCCL in ${_NCCL_HINTS}")
        elseif(WITH_HYGON)
            foreach(_hygon_nccl_env DTKROOT DTK_ROOT ROCM_PATH)
                if(DEFINED ENV{${_hygon_nccl_env}} AND NOT "$ENV{${_hygon_nccl_env}}" STREQUAL "")
                    list(APPEND _NCCL_HINTS "$ENV{${_hygon_nccl_env}}")
                endif()
            endforeach()

            if(DEFINED ENV{HIP_PATH} AND NOT "$ENV{HIP_PATH}" STREQUAL "")
                get_filename_component(_HYGON_NCCL_ROOT_FROM_HIP "$ENV{HIP_PATH}/.." ABSOLUTE)
                list(APPEND _NCCL_HINTS "${_HYGON_NCCL_ROOT_FROM_HIP}")
            endif()

            list(APPEND _NCCL_HINTS /opt/dtk /opt/dtk/rccl)
            set(_NCCL_HEADER_NAMES rccl/rccl.h rccl.h)
            set(_NCCL_LIBRARY_NAMES rccl)
            list(APPEND _NCCL_LIBRARY_SUFFIXES rccl/lib)
            message(STATUS "Hygon detected. Searching for NCCL-compatible RCCL in ${_NCCL_HINTS}")
        endif()

        find_path(AUTO_NCCL_INC NAMES ${_NCCL_HEADER_NAMES}
            HINTS ${_NCCL_HINTS}
            PATH_SUFFIXES ${_NCCL_INCLUDE_SUFFIXES}
            QUIET)
        find_library(AUTO_NCCL_LIB NAMES ${_NCCL_LIBRARY_NAMES}
            HINTS ${_NCCL_HINTS}
            PATH_SUFFIXES ${_NCCL_LIBRARY_SUFFIXES}
            QUIET)

        if(AUTO_NCCL_INC AND AUTO_NCCL_LIB)
            set(WITH_NCCL ON)
            message(STATUS "Auto-detected NCCL backend.")
        else()
            message(STATUS "NCCL-compatible library/headers not found in configured paths.")
        endif()
    else()
        message(STATUS "No suitable device environment, skipping NCCL detection.")
    endif()

    # Detect MCCL Dependencies
    if(WITH_METAX)
        set(_MCCL_HINTS)
        if(DEFINED ENV{MACA_PATH})
            list(APPEND _MCCL_HINTS "$ENV{MACA_PATH}")
        endif()
        list(APPEND _MCCL_HINTS /opt/maca)

        find_path(AUTO_MCCL_INC NAMES mccl.h HINTS ${_MCCL_HINTS} PATH_SUFFIXES include QUIET)
        find_library(AUTO_MCCL_LIB NAMES mccl HINTS ${_MCCL_HINTS} PATH_SUFFIXES lib lib64 QUIET)

        if(AUTO_MCCL_INC AND AUTO_MCCL_LIB)
            set(WITH_MCCL ON)
            message(STATUS "Auto-detected MCCL backend.")
        else()
            message(STATUS "MCCL library/headers not found in MetaX paths.")
        endif()
    elseif(WITH_MOORE)
        set(_MCCL_HINTS)
        foreach(_musa_env MUSA_ROOT MUSA_HOME MUSA_PATH)
            if(DEFINED ENV{${_musa_env}} AND NOT "$ENV{${_musa_env}}" STREQUAL "")
                list(APPEND _MCCL_HINTS "$ENV{${_musa_env}}")
            endif()
        endforeach()
        list(APPEND _MCCL_HINTS /usr/local/musa)

        find_path(AUTO_MCCL_INC NAMES mccl.h HINTS ${_MCCL_HINTS} PATH_SUFFIXES include QUIET)
        find_library(AUTO_MCCL_LIB NAMES mccl HINTS ${_MCCL_HINTS} PATH_SUFFIXES lib lib64 QUIET)

        if(AUTO_MCCL_INC AND AUTO_MCCL_LIB)
            set(WITH_MCCL ON)
            message(STATUS "Auto-detected MCCL backend.")
        else()
            message(STATUS "MCCL library/headers not found in Moore paths.")
        endif()
    else()
        message(STATUS "No suitable device environment, skipping MCCL detection.")
    endif()

    # Detect CNCL Dependencies
    if(WITH_CAMBRICON)
        find_path(AUTO_CNCL_INC NAMES cncl.h HINTS "$ENV{NEUWARE_HOME}" /usr/local/neuware PATH_SUFFIXES include QUIET)
        find_library(AUTO_CNCL_LIB NAMES cncl HINTS "$ENV{NEUWARE_HOME}" /usr/local/neuware PATH_SUFFIXES lib lib64 QUIET)

        if(AUTO_CNCL_INC AND AUTO_CNCL_LIB)
            set(WITH_CNCL ON)
            message(STATUS "Auto-detected CNCL backend.")
        else()
            message(STATUS "CNCL library/headers not found in Cambricon paths.")
        endif()
    endif()
endif()

# Fallback: If no backends are enabled or auto-detected, fall back to OpenMPI as the default bootstrap profile.
if(NOT WITH_OMPI AND NOT WITH_MPICH AND NOT WITH_NCCL AND NOT WITH_MCCL AND NOT WITH_CNCL)
    set(WITH_OMPI ON)
    message(STATUS "No backend specified or detected. Defaulting to `WITH_OMPI=ON`")
endif()

# =========================================================
# --- DEPENDENCY DISCOVERY ---
# =========================================================
if(WITH_THEAD)
    _collect_thead_hints(_THEAD_CUDA_HINTS)
    list(APPEND _THEAD_CUDA_HINTS /usr/local/cuda)
    list(REMOVE_DUPLICATES _THEAD_CUDA_HINTS)

    set(THEAD_CUDA_ROOT "")
    foreach(_thead_cuda_hint ${_THEAD_CUDA_HINTS})
        if(NOT THEAD_CUDA_ROOT AND EXISTS "${_thead_cuda_hint}/include/cuda_runtime.h")
            set(THEAD_CUDA_ROOT "${_thead_cuda_hint}")
        endif()
    endforeach()

    if(NOT THEAD_CUDA_ROOT)
        message(FATAL_ERROR "`WITH_THEAD` is `ON` but the T-Head CUDA-compatible SDK was not found. Set `PPU_SDK`, `PPU_HOME`, `CUDA_SDK`, `CUDA_HOME`, or `CUDA_PATH`.")
    endif()

    set(CUDAToolkit_ROOT "${THEAD_CUDA_ROOT}" CACHE PATH "T-Head CUDA-compatible toolkit root" FORCE)
    list(APPEND _NCCL_HINTS ${_THEAD_CUDA_HINTS})
    list(REMOVE_DUPLICATES _NCCL_HINTS)

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

if(WITH_NVIDIA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

if(WITH_ILUVATAR)
    find_program(ILUVATAR_CUDA_COMPILER NAMES clang++ HINTS /usr/local/corex/bin)
    
    if(NOT ILUVATAR_CUDA_COMPILER)
        message(FATAL_ERROR "`WITH_ILUVATAR` is `ON` but CoreX `clang++` was not found.")
    endif()
    
    get_filename_component(ILUVATAR_CUDA_BIN_DIR "${ILUVATAR_CUDA_COMPILER}" DIRECTORY)
    get_filename_component(ILUVATAR_CUDA_ROOT "${ILUVATAR_CUDA_BIN_DIR}/.." ABSOLUTE)
    
    set(CUDAToolkit_ROOT "${ILUVATAR_CUDA_ROOT}" CACHE PATH "Iluvatar CoreX toolkit root" FORCE)
    find_package(CUDAToolkit REQUIRED)

    set(ILUVATAR_INCLUDE_DIR "${ILUVATAR_CUDA_ROOT}/include")
    set(ILUVATAR_LIB_DIR "${ILUVATAR_CUDA_ROOT}/lib" "${ILUVATAR_CUDA_ROOT}/lib64")

    set(ILUVATAR_CUDA_FLAGS
        "-fPIC;-Wno-error=unused-variable;-Wno-error=unused-private-field;-Wno-unused-variable;-std=c++17;--cuda-path=${ILUVATAR_CUDA_ROOT};-x;ivcore"
        CACHE STRING "Iluvatar CUDA compiler flags")

    message(STATUS "Iluvatar: CUDA compiler ${ILUVATAR_CUDA_COMPILER}, arch ${ILUVATAR_ARCH}, toolkit ${ILUVATAR_CUDA_ROOT}")
endif()

if(WITH_METAX)
    set(MACA_PATH $ENV{MACA_PATH})
    if(NOT MACA_PATH)
        # Normally can be found at: `/opt/maca/`.
        set(MACA_PATH "/opt/maca")
        message(WARNING "MACA_PATH environment variable not set. Defaulting to /opt/maca/. Please verify this path is correct.")
    endif()

    get_filename_component(MXCC_WRAPPER_ABS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/devices/mxcc_wrapper.sh" ABSOLUTE)
    
    set(CMAKE_C_COMPILER   "${MXCC_WRAPPER_ABS}")
    set(CMAKE_CXX_COMPILER "${MXCC_WRAPPER_ABS}")

    include_directories("${MACA_PATH}/include")
    link_directories("${MACA_PATH}/lib")

    find_library(MACA_RUNTIME_LIB NAMES mcruntime HINTS "${MACA_PATH}/lib" REQUIRED)
endif()

if(WITH_MOORE)
    include(cmake/architecture/MusaArchitecture.cmake)

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

    get_filename_component(MCC_WRAPPER_ABS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/devices/mcc_wrapper.sh" ABSOLUTE)
    file(CHMOD "${MCC_WRAPPER_ABS}"
     PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
    set(CMAKE_C_COMPILER   "${MCC_WRAPPER_ABS}")
    set(CMAKE_CXX_COMPILER "${MCC_WRAPPER_ABS}")

    include_directories("${MUSA_ROOT}/include")
    link_directories("${MUSA_ROOT}/lib")

    find_library(MUSA_LIB   NAMES musa   HINTS "${MUSA_ROOT}/lib" REQUIRED)
    find_library(MUSART_LIB NAMES musart HINTS "${MUSA_ROOT}/lib" REQUIRED)
    find_library(MUBLAS_LIB NAMES mublas HINTS "${MUSA_ROOT}/lib" REQUIRED)

    infiniccl_resolve_musa_architecture_config(
        "${MUSA_ROOT}/include"
        "${MUSART_LIB}"
    )
endif()

if(WITH_CAMBRICON)
    set(NEUWARE_HOME $ENV{NEUWARE_HOME})

    include_directories("${NEUWARE_HOME}/include")
    link_directories("${NEUWARE_HOME}/lib")
    link_directories("${NEUWARE_HOME}/lib64")

    # Libraries: `cnrt` / `cnnl` / `cnnl_extra` / `cnpapi`.
    find_library(CAMBRICON_RUNTIME_LIB    NAMES cnrt       HINTS "${NEUWARE_HOME}/lib64" REQUIRED)
    find_library(CAMBRICON_CNNL_LIB       NAMES cnnl       HINTS "${NEUWARE_HOME}/lib64" REQUIRED)
    find_library(CAMBRICON_CNNL_EXTRA_LIB NAMES cnnl_extra HINTS "${NEUWARE_HOME}/lib64" REQUIRED)
    find_library(CAMBRICON_PAPI_LIB       NAMES cnpapi     HINTS "${NEUWARE_HOME}/lib64" REQUIRED)
endif()

if(WITH_HYGON)
    set(HYGON_DTK_ROOT "")
    foreach(_hygon_env DTKROOT DTK_ROOT ROCM_PATH)
        if(NOT HYGON_DTK_ROOT AND DEFINED ENV{${_hygon_env}} AND NOT "$ENV{${_hygon_env}}" STREQUAL "")
            set(HYGON_DTK_ROOT "$ENV{${_hygon_env}}")
        endif()
    endforeach()

    if(NOT HYGON_DTK_ROOT AND DEFINED ENV{HIP_PATH} AND NOT "$ENV{HIP_PATH}" STREQUAL "")
        get_filename_component(HYGON_DTK_ROOT "$ENV{HIP_PATH}/.." ABSOLUTE)
    endif()

    if(NOT HYGON_DTK_ROOT AND EXISTS "/opt/dtk")
        set(HYGON_DTK_ROOT "/opt/dtk")
    endif()

    if(NOT HYGON_DTK_ROOT)
        message(FATAL_ERROR "`WITH_HYGON` is `ON` but DTK was not found. Set `DTKROOT`, `DTK_ROOT`, `ROCM_PATH`, or `HIP_PATH`.")
    endif()

    set(HYGON_HIP_HINTS "${HYGON_DTK_ROOT}" "${HYGON_DTK_ROOT}/hip")
    find_path(HYGON_HIP_INCLUDE_DIR
        NAMES hip/hip_runtime.h
        HINTS ${HYGON_HIP_HINTS}
        PATH_SUFFIXES include hip/include
        REQUIRED
    )
    find_library(HYGON_HIP_RUNTIME_LIB
        NAMES amdhip64
        HINTS ${HYGON_HIP_HINTS}
        PATH_SUFFIXES lib lib64 hip/lib
        REQUIRED
    )
    find_program(HYGON_HIPCC_COMPILER
        NAMES hipcc
        HINTS ${HYGON_HIP_HINTS}
        PATH_SUFFIXES bin hip/bin
        REQUIRED
    )

    set(CMAKE_CXX_COMPILER "${HYGON_HIPCC_COMPILER}" CACHE FILEPATH "Hygon DTK hipcc compiler" FORCE)
    set(HYGON_HIP_DEFINITIONS __HIP_PLATFORM_AMD__)
    include_directories(${HYGON_HIP_INCLUDE_DIR})
endif()

if(WITH_OMPI OR WITH_MPICH)
    find_package(MPI REQUIRED)
endif()

if(WITH_NCCL)
    if (NOT WITH_NVIDIA AND NOT WITH_THEAD AND NOT WITH_ILUVATAR AND NOT WITH_HYGON)
        message(FATAL_ERROR "NCCL backend requires NVIDIA, T-Head, Iluvatar, or Hygon GPU support. Please enable `WITH_NVIDIA`, `WITH_THEAD`, `WITH_ILUVATAR`, or `WITH_HYGON`.")
    endif()

    set(_NCCL_HEADER_NAMES nccl.h)
    set(_NCCL_LIBRARY_NAMES nccl)
    set(NCCL_COMPILE_DEFINITIONS)
    set(_NCCL_INCLUDE_SUFFIXES include)
    set(_NCCL_LIBRARY_SUFFIXES lib lib64)

    if(WITH_THEAD)
        set(_NCCL_HINTS ${_THEAD_CUDA_HINTS})
        list(APPEND _NCCL_INCLUDE_SUFFIXES targets/x86_64-linux/include)
        list(APPEND _NCCL_LIBRARY_SUFFIXES targets/x86_64-linux/lib)
    elseif(WITH_HYGON)
        list(APPEND _NCCL_HINTS "${HYGON_DTK_ROOT}" "${HYGON_DTK_ROOT}/rccl")
        set(_NCCL_HEADER_NAMES rccl/rccl.h rccl.h)
        set(_NCCL_LIBRARY_NAMES rccl)
        set(NCCL_COMPILE_DEFINITIONS INFINI_CCL_USE_RCCL)
        list(APPEND _NCCL_LIBRARY_SUFFIXES rccl/lib)
    endif()

    list(REMOVE_DUPLICATES _NCCL_HINTS)
    find_library(NCCL_LIB NAMES ${_NCCL_LIBRARY_NAMES}
        HINTS ${_NCCL_HINTS}
        PATH_SUFFIXES ${_NCCL_LIBRARY_SUFFIXES}
        REQUIRED)
    find_path(NCCL_INC NAMES ${_NCCL_HEADER_NAMES}
        HINTS ${_NCCL_HINTS}
        PATH_SUFFIXES ${_NCCL_INCLUDE_SUFFIXES}
        REQUIRED)

    include_directories(${NCCL_INC})
endif()

if(WITH_MCCL)
    if (NOT WITH_METAX AND NOT WITH_MOORE)
        message(FATAL_ERROR "MCCL backend requires MetaX or Moore GPU support. Please enable `WITH_METAX` or `WITH_MOORE`.")
    endif()

    set(_MCCL_HINTS)
    if(MACA_PATH)
        list(APPEND _MCCL_HINTS "${MACA_PATH}")
    endif()
    if(DEFINED ENV{MACA_PATH})
        list(APPEND _MCCL_HINTS "$ENV{MACA_PATH}")
    endif()
    list(APPEND _MCCL_HINTS /opt/maca)

    if(WITH_MOORE)
        set(_MCCL_HINTS "${MUSA_ROOT}" /usr/local/musa)
    endif()

    find_library(MCCL_LIB NAMES mccl HINTS ${_MCCL_HINTS} PATH_SUFFIXES lib lib64 REQUIRED)
    find_path(MCCL_INC NAMES mccl.h HINTS ${_MCCL_HINTS} PATH_SUFFIXES include REQUIRED)

    include_directories(${MCCL_INC})
endif()

if(WITH_CNCL)
    if(NOT WITH_CAMBRICON)
        message(FATAL_ERROR "CNCL backend requires Cambricon device support. Please enable `WITH_CAMBRICON`.")
    endif()

    find_library(CNCL_LIB NAMES cncl HINTS "${NEUWARE_HOME}" "$ENV{NEUWARE_HOME}" /usr/local/neuware PATH_SUFFIXES lib lib64 REQUIRED)
    find_path(CNCL_INC NAMES cncl.h HINTS "${NEUWARE_HOME}" "$ENV{NEUWARE_HOME}" /usr/local/neuware PATH_SUFFIXES include REQUIRED)

    include_directories(${CNCL_INC})
endif()

# Python is required for code generation.
find_package(Python3 REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(src)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# =========================================================
# --- Installation ---
# =========================================================
# For Development: Copy launcher to `build/`.
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/icclrun.py"
    "${CMAKE_BINARY_DIR}/icclrun"
    COPYONLY
)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/icclrun_logic.py"
    "${CMAKE_BINARY_DIR}/icclrun_logic.py"
    COPYONLY
)

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/scripts/backends" DESTINATION "${CMAKE_BINARY_DIR}")

# Ensure it is executable in `build/`.
file(CHMOD "${CMAKE_BINARY_DIR}/icclrun" 
     PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

# For Production/System Install
include(GNUInstallDirs)

# Install the library and record it in an export set so that downstream
# projects can pull in `InfiniCCL::infiniccl` via `find_package(InfiniCCL)`.
# All hardware/backend dependencies (CUDA, MPI, NCCL, ...) are linked PRIVATE
# into this shared library, so they are NOT propagated as usage requirements.
install(TARGETS infiniccl
    EXPORT InfiniCCLTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/infiniccl)

install(PROGRAMS scripts/icclrun.py DESTINATION ${CMAKE_INSTALL_BINDIR} RENAME icclrun)

# =========================================================
# --- CMake Package Config (`find_package(InfiniCCL)`) ---
# =========================================================
set(INFINICCL_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/InfiniCCL)

# Export the imported target definition (`InfiniCCL::infiniccl`).
install(EXPORT InfiniCCLTargets
    FILE InfiniCCLTargets.cmake
    NAMESPACE InfiniCCL::
    DESTINATION ${INFINICCL_CMAKE_DIR}
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/InfiniCCLConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/InfiniCCLConfig.cmake"
    INSTALL_DESTINATION ${INFINICCL_CMAKE_DIR}
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/InfiniCCLConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/InfiniCCLConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/InfiniCCLConfigVersion.cmake"
    DESTINATION ${INFINICCL_CMAKE_DIR}
)

# Install the supporting logic to a private folder.
set(ICCL_PRIVATE_LIB ${CMAKE_INSTALL_LIBDIR}/infiniccl)
install(FILES scripts/icclrun_logic.py DESTINATION ${ICCL_PRIVATE_LIB})
install(DIRECTORY scripts/backends DESTINATION ${ICCL_PRIVATE_LIB})
