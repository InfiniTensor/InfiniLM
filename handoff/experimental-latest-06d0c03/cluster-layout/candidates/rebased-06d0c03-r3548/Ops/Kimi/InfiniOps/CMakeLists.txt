cmake_minimum_required(VERSION 3.18)
if(SKBUILD_STATE STREQUAL "sdist")
    # Source packaging must not probe compilers, GPUs or installed SDKs.
    project(InfiniOps LANGUAGES NONE)
    include(cmake/MarlinSource.cmake)
    foreach(_marlin_file IN LISTS _marlin_source_files)
        configure_file("${vllm_marlin_SOURCE_DIR}/${_marlin_file}"
            "${CMAKE_BINARY_DIR}/marlin-source/${_marlin_file}" COPYONLY)
    endforeach()
    include(cmake/FlashAttentionSource.cmake)
    foreach(_flash_file IN LISTS _flash_source_files)
        configure_file("${flash_attention_SOURCE_DIR}/${_flash_file}"
            "${CMAKE_BINARY_DIR}/flash-attention-source/${_flash_file}" COPYONLY)
    endforeach()
    configure_file("${flash_attention_cutlass_SOURCE_DIR}/LICENSE.txt"
        "${CMAKE_BINARY_DIR}/flash-attention-cutlass-source/LICENSE.txt" COPYONLY)
    file(COPY "${flash_attention_cutlass_SOURCE_DIR}/include"
        DESTINATION "${CMAKE_BINARY_DIR}/flash-attention-cutlass-source")
    return()
endif()
project(InfiniOps LANGUAGES CXX)

if(POLICY CMP0116)
    cmake_policy(SET CMP0116 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Internal variable to control `pybind11`'s automatic optimization flags (like `-flto`).
set(PYBIND11_ENABLE_EXTRAS ON)

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

option(WITH_TORCH "Enable PyTorch C++ backend" OFF)
option(WITH_LINKED "Enable linked third-party operator backends" OFF)

option(BUILD_NVIDIA_FLASH_INFERENCE "Build pinned Torch-free NVIDIA inference Attention" OFF)
option(BUILD_NVIDIA_W4_MARLIN "Build NVIDIA AWQ/GPTQ Marlin providers (CUDA >= 12.3)" OFF)

option(WITH_NINETOOTHED "Enable NineToothed-generated kernels" OFF)

option(WITH_TRITON "Enable the Triton JIT backend" OFF)

# Custom `AscendC` kernels under `src/native/ascend/custom/`. `ON` by default
# so CI and routine dev builds always exercise `implementation_index=1/2`
# for `RmsNorm` / `AddRmsNorm`. Gated by `WITH_ASCEND` in
# `src/CMakeLists.txt`, so non-Ascend builds ignore it. Pass
# `-DBUILD_ASCEND_CUSTOM=OFF` to skip the `ccec` build on Ascend
# machines where the custom kernels aren't needed.
#
# When `ON`, `src/CMakeLists.txt` drives the standalone
# `src/native/ascend/custom/build.sh` via a build-phase custom command. This
# sidesteps a `CANN` `extract_host_stub.py` path bug that breaks in-tree
# `ascendc_library()` under `scikit-build-core` temp-dir builds, then links
# the produced `libno_workspace_kernel.a` into the `ops` module with
# `--whole-archive`. Requires `torch_npu` and the `AscendC` toolchain (`ccec`).
option(BUILD_ASCEND_CUSTOM "Build custom AscendC kernels" ON)

option(AUTO_DETECT_DEVICES "Automatically detect available devices" OFF)
option(AUTO_DETECT_BACKENDS "Automatically detect available backends" OFF)
option(GENERATE_OPERATOR_CALL_INSTANTIATIONS
    "Generate explicit operator call instantiations" ON)
option(GENERATE_PYTHON_BINDINGS "Generate Python bindings" OFF)
option(INFINI_OPS_BUILD_DOCS "Build InfiniOps documentation" OFF)
option(INFINI_OPS_ENABLE_HOST_RANGE_PROFILING
    "Enable host-side range profiling" OFF)

set(_DEFAULT_HYGON_DTK_ROOT "/opt/dtk")

function(_infini_ops_find_hygon_cuda_root out_var dtk_root)
    set(_candidates
        "${dtk_root}/cuda"
        "${dtk_root}/cuda/cuda"
    )

    file(GLOB _versioned_cuda_dirs LIST_DIRECTORIES true "${dtk_root}/cuda/cuda-*")
    if(_versioned_cuda_dirs)
        list(SORT _versioned_cuda_dirs)
        list(REVERSE _versioned_cuda_dirs)
        list(APPEND _candidates ${_versioned_cuda_dirs})
    endif()

    foreach(_candidate IN LISTS _candidates)
        if(EXISTS "${_candidate}/bin/nvcc")
            set(${out_var} "${_candidate}" PARENT_SCOPE)
            return()
        endif()
    endforeach()

    set(${out_var} "" PARENT_SCOPE)
endfunction()

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

    set(_hygon_detected FALSE)
    if(DEFINED ENV{DTK_ROOT} AND NOT "$ENV{DTK_ROOT}" STREQUAL "")
        _infini_ops_find_hygon_cuda_root(_HYGON_CUDA_DETECT_ROOT "$ENV{DTK_ROOT}")
        if(_HYGON_CUDA_DETECT_ROOT)
            set(_hygon_detected TRUE)
        endif()
    else()
        _infini_ops_find_hygon_cuda_root(_HYGON_CUDA_DETECT_ROOT "${_DEFAULT_HYGON_DTK_ROOT}")
        if(_HYGON_CUDA_DETECT_ROOT)
            set(_hygon_detected TRUE)
        endif()
    endif()

    if(_hygon_detected)
        set(WITH_HYGON ON)
        message(STATUS "Auto-detected Hygon environment.")
    endif()

    if(DEFINED ENV{CUDA_SDK} AND NOT "$ENV{CUDA_SDK}" STREQUAL "")
        set(_thead_cuda_detect_root "$ENV{CUDA_SDK}")
    else()
        set(_thead_cuda_detect_root "/usr/local/PPU_SDK/CUDA_SDK")
    endif()

    file(GLOB _thead_ppu_devices "/dev/alixpu_ppu*")

    if(_thead_ppu_devices
    AND EXISTS "${_thead_cuda_detect_root}/bin/nvcc")
        set(WITH_THEAD ON)
        message(STATUS "Auto-detected T-Head PPU environment.")
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

if(BUILD_NVIDIA_FLASH_INFERENCE AND NOT WITH_NVIDIA)
    message(FATAL_ERROR "Native FlashAttention requires WITH_NVIDIA")
endif()
if(BUILD_NVIDIA_W4_MARLIN AND NOT WITH_NVIDIA)
    message(FATAL_ERROR "Marlin requires WITH_NVIDIA")
endif()

if(AUTO_DETECT_BACKENDS)
    message(STATUS "Auto-detecting available backends...")

    # The Python that scikit-build's build-isolated environment hands
    # us does not have `torch` (only `[build-system].requires` is
    # installed).  Fall back to a list of common system interpreters so
    # the auto-detection finds `torch` when it is in the install env
    # but not the build env.  The first interpreter that successfully
    # imports `torch` wins and is reused by the `WITH_TORCH` block
    # below for include / library lookups.
    find_package(Python COMPONENTS Interpreter QUIET)

    set(_torch_python_candidates "${Python_EXECUTABLE}")
    foreach(_candidate
            python3
            python
            /usr/bin/python3
            /usr/local/bin/python3
            /opt/conda/bin/python
            /opt/conda/bin/python3)
        find_program(_resolved_${_candidate} ${_candidate})
        if(_resolved_${_candidate} AND
           NOT _resolved_${_candidate} STREQUAL "${Python_EXECUTABLE}")
            list(APPEND _torch_python_candidates "${_resolved_${_candidate}}")
        endif()
    endforeach()

    foreach(_py ${_torch_python_candidates})
        if(NOT _py)
            continue()
        endif()

        execute_process(
            COMMAND "${_py}" -c "import torch"
            RESULT_VARIABLE _torch_import_result
            OUTPUT_QUIET
            ERROR_QUIET
        )

        if(_torch_import_result EQUAL 0)
            set(_TORCH_PYTHON "${_py}")
            break()
        endif()
    endforeach()

    if(_TORCH_PYTHON)
        set(WITH_TORCH ON)
        message(STATUS "Auto-detected PyTorch (via ${_TORCH_PYTHON}).")
    endif()
endif()

 # A linked backend is not necessarily a Torch backend: TVM-FFI transports
 # consume DLPack views and their own runtime. Discover Torch lazily only when
 # native Torch wrappers or a selected linked Torch transport need it.
macro(infiniops_find_torch)
    if(WITH_MARS AND WITH_TORCH AND WITH_LINKED)
        find_package(Python COMPONENTS Interpreter Development.Embed REQUIRED)
    else()
        find_package(Python COMPONENTS Interpreter REQUIRED)
    endif()

    # Prefer the interpreter that the auto-detect block already
    # confirmed has `torch` (this is the system Python on hosts that
    # use scikit-build's build-isolation, where the build interpreter
    # does not have `torch`).  Fall back to `Python_EXECUTABLE` for
    # explicit `-DWITH_TORCH=ON` invocations.
    if(NOT _TORCH_PYTHON)
        set(_TORCH_PYTHON "${Python_EXECUTABLE}")
    endif()

    # Query `torch` paths directly instead of using `find_package(Torch)`,
    # which pulls in Caffe2's CMake config and may fail on platforms with
    # non-standard CUDA toolchains.
    execute_process(
        COMMAND ${_TORCH_PYTHON} -c "from torch.utils.cpp_extension import include_paths; print(';'.join(include_paths()))"
        OUTPUT_VARIABLE TORCH_INCLUDE_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _torch_result
    )

    if(NOT _torch_result EQUAL 0)
        message(FATAL_ERROR
            "The selected Torch backend/transport requires an installed `torch` package.")
    endif()

    execute_process(
        COMMAND ${_TORCH_PYTHON} -c "from torch.utils.cpp_extension import library_paths; print(';'.join(library_paths()))"
        OUTPUT_VARIABLE _torch_lib_dirs
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    execute_process(
        COMMAND ${_TORCH_PYTHON} -c "import pathlib, torch; p = pathlib.Path(torch.__file__).resolve().parent.parent / 'torch.libs'; print(str(p) if p.exists() else '')"
        OUTPUT_VARIABLE _torch_private_lib_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    set(TORCH_RUNTIME_DIRS ${_torch_lib_dirs})
    if(_torch_private_lib_dir)
        list(APPEND TORCH_RUNTIME_DIRS ${_torch_private_lib_dir})
    endif()

    find_library(TORCH_LIB torch HINTS ${_torch_lib_dirs} REQUIRED)
    find_library(TORCH_CPU_LIB torch_cpu HINTS ${_torch_lib_dirs} REQUIRED)
    find_library(C10_LIB c10 HINTS ${_torch_lib_dirs} REQUIRED)
    set(TORCH_LIBRARIES ${TORCH_LIB} ${TORCH_CPU_LIB} ${C10_LIB})

    if(WITH_NVIDIA OR WITH_ILUVATAR OR WITH_THEAD OR WITH_MARS)
        find_library(TORCH_CUDA_LIB torch_cuda HINTS ${_torch_lib_dirs} REQUIRED)
        find_library(C10_CUDA_LIB c10_cuda HINTS ${_torch_lib_dirs} REQUIRED)
        # `torch_cuda` registers CUDA kernels through static initializers, so
        # retain it even though ATen dispatcher calls do not reference it directly.
        list(APPEND TORCH_LIBRARIES
            -Wl,--no-as-needed ${TORCH_CUDA_LIB} ${C10_CUDA_LIB} -Wl,--as-needed)
    endif()

    if(WITH_HYGON)
        find_library(TORCH_HIP_LIB torch_hip HINTS ${_torch_lib_dirs} REQUIRED)
        find_library(C10_HIP_LIB c10_hip HINTS ${_torch_lib_dirs} REQUIRED)
        list(APPEND TORCH_LIBRARIES
            -Wl,--no-as-needed ${TORCH_HIP_LIB} ${C10_HIP_LIB} -Wl,--as-needed)
    endif()

    if(WITH_MARS AND WITH_TORCH AND WITH_LINKED)
        # The Mars FlashAttention wheel loads libtorch_python without declaring
        # its libpython dependency, so keep the embedding library in our chain.
        list(APPEND TORCH_LIBRARIES Python::Python)
    endif()

    if(WITH_METAX)
        find_library(C10_CUDA_LIB c10_cuda HINTS ${_torch_lib_dirs} REQUIRED)
        find_library(MACA_TORCH_RUNTIME_LIB runtime_cu
            HINTS "$ENV{MACA_PATH}/lib" REQUIRED)
        list(APPEND TORCH_LIBRARIES
            ${C10_CUDA_LIB} ${MACA_TORCH_RUNTIME_LIB})
    endif()

    # `auditwheel`-repaired `torch` wheels bundle transitive dependencies
    # (e.g. `libgfortran-<hash>.so`, `libopenblasp-<hash>.so`) in a sibling
    # `torch.libs/` directory that `library_paths()` does not return.  When
    # building against such a wheel, the linker needs this path to resolve
    # the bundled `NEEDED` entries (otherwise: `undefined reference to
    # _gfortran_etime@GFORTRAN_8` etc.).
    execute_process(
        COMMAND ${_TORCH_PYTHON} -c "import os, torch; d = os.path.dirname(torch.__file__); p = os.path.join(os.path.dirname(d), 'torch.libs'); print(p if os.path.isdir(p) else '')"
        OUTPUT_VARIABLE TORCH_BUNDLED_LIBS_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(TORCH_BUNDLED_LIBS_DIR)
        list(APPEND CMAKE_BUILD_RPATH "${TORCH_BUNDLED_LIBS_DIR}")
        list(APPEND CMAKE_INSTALL_RPATH "${TORCH_BUNDLED_LIBS_DIR}")
        # `rpath-link` is linker-only: lets `ld` resolve the bundled
        # transitive `NEEDED` entries at link time without adding them to our
        # own binary's direct `NEEDED` list.
        add_link_options("-Wl,-rpath-link,${TORCH_BUNDLED_LIBS_DIR}")
        message(STATUS "PyTorch bundled libs: ${TORCH_BUNDLED_LIBS_DIR}")
    endif()

    # Query the `CXX11` ABI setting that `torch` was compiled with.
    # A mismatch causes linker errors (e.g. undefined reference to
    # `c10::Device::Device(std::string const&)`).
    execute_process(
        COMMAND ${_TORCH_PYTHON} -c "import torch; print(int(torch.compiled_with_cxx11_abi()))"
        OUTPUT_VARIABLE TORCH_CXX11_ABI
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _torch_abi_result
    )

    if(_torch_abi_result EQUAL 0)
        add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})
        message(STATUS "PyTorch CXX11 ABI: ${TORCH_CXX11_ABI}")
    endif()

    message(STATUS "Found PyTorch: ${TORCH_INCLUDE_DIRS}")
endmacro()

if(WITH_TORCH)
    infiniops_find_torch()
endif()

# Only one CUDA-like GPU backend can be enabled at a time.
set(_gpu_backend_count 0)
foreach(_gpu_backend WITH_NVIDIA WITH_ILUVATAR WITH_HYGON WITH_THEAD WITH_METAX WITH_MARS WITH_MOORE WITH_ASCEND)
    if(${_gpu_backend})
        math(EXPR _gpu_backend_count "${_gpu_backend_count} + 1")
    endif()
endforeach()

if(_gpu_backend_count GREATER 1)
    message(FATAL_ERROR "`WITH_NVIDIA`, `WITH_ILUVATAR`, `WITH_HYGON`, `WITH_THEAD`, `WITH_METAX`, `WITH_MARS`, `WITH_MOORE`, and `WITH_ASCEND` are mutually exclusive. Build one GPU backend at a time.")
endif()

if(WITH_NINETOOTHED AND NOT WITH_NVIDIA)
    message(FATAL_ERROR "`WITH_NINETOOTHED` currently requires `WITH_NVIDIA=ON` because NineToothed AOT uses `caller=\"cuda\"`.")
endif()

if(WITH_NINETOOTHED)
    set(NINETOOTHED_PYTHON_EXECUTABLE "" CACHE FILEPATH "Python executable used to run NineToothed code generation")
endif()

if(WITH_TRITON)
    if(NOT WITH_NVIDIA)
        message(
            FATAL_ERROR
            "`WITH_TRITON` requires `WITH_NVIDIA=ON` because NVIDIA is the only implemented Triton JIT backend."
        )
    endif()
    if(NOT GENERATE_PYTHON_BINDINGS)
        message(
            FATAL_ERROR
            "`WITH_TRITON` requires `GENERATE_PYTHON_BINDINGS=ON` because the runtime compiler is shipped in the Python package."
        )
    endif()
endif()

if(WITH_NVIDIA)
    add_compile_definitions(WITH_NVIDIA=1)
    enable_language(CUDA)
    if(WITH_TRITON)
        find_package(CUDAToolkit 12.0 REQUIRED)
    else()
        find_package(CUDAToolkit REQUIRED)
    endif()
endif()

# Iluvatar: CUDA-compatible device. CoreX clang++ works for `-x ivcore`
# compilation, but CMake's CUDA language path appends `-x cuda` and can crash
# the compiler. Keep CMake's CUDA language disabled for this backend and
# compile the few generated dispatch sources that need CUDA syntax via custom
# clang++ commands.
# Reference: `InfiniCore` `xmake/iluvatar.lua`.
if(WITH_ILUVATAR)
    add_compile_definitions(WITH_ILUVATAR=1 __ILUVATAR__=1)
    set(ILUVATAR_ARCH "ivcore20" CACHE STRING "Iluvatar GPU architecture")
    find_program(ILUVATAR_CUDA_COMPILER NAMES clang++ HINTS /usr/local/corex/bin)
    if(NOT ILUVATAR_CUDA_COMPILER)
        message(FATAL_ERROR "`WITH_ILUVATAR` is `ON` but CoreX `clang++` was not found.")
    endif()
    set(ILUVATAR_CUDA_FLAGS
        "--cuda-gpu-arch=${ILUVATAR_ARCH};-fPIC;-Wno-error=unused-variable;-Wno-error=unused-private-field;-Wno-unused-variable;-std=c++17;--cuda-path=/usr/local/corex;-x;ivcore"
        CACHE STRING "Iluvatar CUDA compiler flags")
    message(STATUS "Iluvatar: CUDA compiler ${ILUVATAR_CUDA_COMPILER}, arch ${ILUVATAR_ARCH}")
    find_package(CUDAToolkit REQUIRED)
endif()

if(WITH_HYGON)
    add_compile_definitions(WITH_HYGON=1)
    set(DTK_ROOT $ENV{DTK_ROOT})
    if(NOT DTK_ROOT)
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
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            string(REGEX MATCH "gfx[0-9]+" _HYGON_ARCH_AUTO "${_HYGON_ROCMINFO_OUTPUT}")
            if(_HYGON_ARCH_AUTO)
                set(_HYGON_ARCH_DEFAULT "${_HYGON_ARCH_AUTO}")
            endif()
        endif()
    endif()

    set(HYGON_ARCH "${_HYGON_ARCH_DEFAULT}" CACHE STRING "Hygon GPU architecture")
    _infini_ops_find_hygon_cuda_root(HYGON_CUDA_ROOT "${DTK_ROOT}")

    if(NOT HYGON_CUDA_ROOT)
        message(FATAL_ERROR "`WITH_HYGON` is `ON` but no DTK `nvcc` was found under `${DTK_ROOT}`. Checked `${DTK_ROOT}/cuda/bin/nvcc`, `${DTK_ROOT}/cuda/cuda/bin/nvcc`, and `${DTK_ROOT}/cuda/cuda-*/bin/nvcc`.")
    endif()

    set(CMAKE_CUDA_COMPILER "${HYGON_CUDA_ROOT}/bin/nvcc" CACHE FILEPATH "Hygon CUDA compiler (DTK nvcc)")
    set(CUDAToolkit_ROOT "${HYGON_CUDA_ROOT}" CACHE PATH "Hygon CUDA toolkit root")
    set(CMAKE_CUDA_ARCHITECTURES OFF CACHE STRING "Disable default CUDA arch flags for Hygon" FORCE)
    set(CMAKE_CUDA_FLAGS "-std=c++17 -fPIC -arch=${HYGON_ARCH} -Wno-return-type -Wno-error=unused-private-field" CACHE STRING "Hygon CUDA flags")
    set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF CACHE BOOL "Disable RDC for Hygon")

    # DTK's nvcc wrapper derives its toolkit root from `CUDA_PATH`.
    set(ENV{CUDA_PATH} "${HYGON_CUDA_ROOT}")
    set(ENV{CUDA_HOME} "${HYGON_CUDA_ROOT}")

    # DTK's nvcc wrapper may invoke `nvcc` by name during compiler checks.
    set(ENV{PATH} "${HYGON_CUDA_ROOT}/bin:$ENV{PATH}")

    # The actual Ninja build runs in fresh processes. Keep a launcher command
    # for CUDA-backed Python bindings that need the DTK wrapper environment.
    set(_HYGON_RULE_LAUNCH_ENV
        "${CMAKE_COMMAND} -E env CUDA_PATH=${HYGON_CUDA_ROOT} CUDA_HOME=${HYGON_CUDA_ROOT} PATH=${HYGON_CUDA_ROOT}/bin:$ENV{PATH}")

    include_directories("${DTK_ROOT}/include")
    include_directories("${HYGON_CUDA_ROOT}/include")
    link_directories("${DTK_ROOT}/lib")
    link_directories("${HYGON_CUDA_ROOT}/lib64")

    message(STATUS "Hygon: CUDA compiler ${CMAKE_CUDA_COMPILER}, arch ${HYGON_ARCH}, DTK root ${DTK_ROOT}")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

if(WITH_THEAD)
    add_compile_definitions(WITH_THEAD=1)

    if(DEFINED ENV{CUDA_SDK} AND NOT "$ENV{CUDA_SDK}" STREQUAL "")
        set(_thead_cuda_root "$ENV{CUDA_SDK}")
    else()
        set(_thead_cuda_root "/usr/local/PPU_SDK/CUDA_SDK")
    endif()

    if(NOT EXISTS "${_thead_cuda_root}/bin/nvcc")
        message(FATAL_ERROR
            "`WITH_THEAD` is `ON` but the T-Head PPU CUDA SDK was not found. "
            "Set `CUDA_SDK` or install it at `/usr/local/PPU_SDK/CUDA_SDK`.")
    endif()

    set(CMAKE_CUDA_COMPILER "${_thead_cuda_root}/bin/nvcc"
        CACHE FILEPATH "T-Head PPU CUDA compiler" FORCE)
    set(CUDAToolkit_ROOT "${_thead_cuda_root}"
        CACHE PATH "T-Head PPU CUDA toolkit root" FORCE)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING
            "T-Head PPU CUDA architecture" FORCE)
    endif()

    set(ENV{CUDA_PATH} "${_thead_cuda_root}")
    set(ENV{CUDA_HOME} "${_thead_cuda_root}")
    set(ENV{PATH} "${_thead_cuda_root}/bin:$ENV{PATH}")
    include_directories("${_thead_cuda_root}/include")
    link_directories("${_thead_cuda_root}/lib64")

    message(STATUS "T-Head PPU: CUDA toolkit ${_thead_cuda_root}")
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
    find_library(MACA_DNN_LIB     NAMES mcdnn     HINTS "${MACA_PATH}/lib" REQUIRED)
    find_library(MACA_BLAS_LIB    NAMES mcblas    HINTS "${MACA_PATH}/lib" REQUIRED)
endif()

if(WITH_MARS)
    add_compile_definitions(WITH_MARS=1)

    # Normally can be found at: `/opt/hpcc/`.
    set(HPCC_PATH $ENV{HPCC_PATH})
    if(NOT HPCC_PATH)
        set(HPCC_PATH "/opt/hpcc")
    endif()
    set(CMAKE_C_COMPILER   ${CMAKE_CURRENT_SOURCE_DIR}/scripts/htcc_wrapper.sh)
    set(CMAKE_CXX_COMPILER ${CMAKE_CURRENT_SOURCE_DIR}/scripts/htcc_wrapper.sh)

    include_directories("${HPCC_PATH}/include")
    include_directories("${HPCC_PATH}/tools/cu-bridge/include")
    link_directories("${HPCC_PATH}/lib")
    link_directories("${HPCC_PATH}/htgpu_llvm/lib")

    # Libraries: hcruntime / hcdnn / hcblas. HPCC is the toolkit; Mars is the device.
    find_library(HPCC_RUNTIME_LIB NAMES hcruntime HINTS "${HPCC_PATH}/lib" REQUIRED)
    find_library(HPCC_DNN_LIB     NAMES hcdnn     HINTS "${HPCC_PATH}/lib" REQUIRED)
    find_library(HPCC_BLAS_LIB    NAMES hcblas    HINTS "${HPCC_PATH}/lib" REQUIRED)
    # htcc emits LLVM OpenMP (__kmpc_*), not GNU libgomp.
    find_library(HPCC_OMP_LIB NAMES omp iomp5 HINTS "${HPCC_PATH}/htgpu_llvm/lib" REQUIRED)
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
    find_library(MUBLAS_LIB NAMES mublas HINTS "${MUSA_ROOT}/lib" REQUIRED)
    find_library(MUSA_OPENMP_LIB NAMES iomp5 omp HINTS "${MUSA_ROOT}/lib" REQUIRED)
endif()

if(WITH_CAMBRICON)
    add_compile_definitions(WITH_CAMBRICON=1)
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
endif()

if(WITH_TORCH OR WITH_METAX OR WITH_MARS OR WITH_MOORE)
    set(PYBIND11_ENABLE_EXTRAS OFF)
endif()

if(WITH_HYGON AND NOT EXISTS "${DTK_ROOT}/llvm/lib/LLVMgold.so")
    set(PYBIND11_ENABLE_EXTRAS OFF)
endif()

set(INFINI_RT_ROOT "$ENV{INFINI_RT_ROOT}" CACHE PATH "Installed InfiniRT prefix")
set(INFINI_RT_SOURCE_DIR "" CACHE PATH "InfiniRT source directory for development builds")
set(INFINI_RT_INCLUDE_DIRS "" CACHE STRING "InfiniRT include directories")
set(INFINI_RT_LIBRARY "" CACHE FILEPATH "InfiniRT library")

function(_infiniops_configure_infinirt)
    if(TARGET infinirt)
        get_target_property(_infinirt_include_dirs infinirt INTERFACE_INCLUDE_DIRECTORIES)
        if(_infinirt_include_dirs)
            set(INFINI_RT_INCLUDE_DIRS "${_infinirt_include_dirs}" CACHE STRING "InfiniRT include directories" FORCE)
        endif()
        return()
    endif()

    if(INFINI_RT_SOURCE_DIR)
        if(NOT EXISTS "${INFINI_RT_SOURCE_DIR}/CMakeLists.txt")
            message(FATAL_ERROR
                "InfiniRT source tree not found at `${INFINI_RT_SOURCE_DIR}`. "
                "Unset `INFINI_RT_SOURCE_DIR` to use an installed InfiniRT package.")
        endif()

        add_subdirectory("${INFINI_RT_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/InfiniRT")
        set(INFINI_RT_INCLUDE_DIRS
            "${INFINI_RT_SOURCE_DIR}/include;${INFINI_RT_SOURCE_DIR}/generated/include"
            CACHE STRING "InfiniRT include directories" FORCE)
        return()
    endif()

    set(_infinirt_include_hints ${INFINI_RT_INCLUDE_DIRS})
    if(INFINI_RT_ROOT)
        list(APPEND _infinirt_include_hints "${INFINI_RT_ROOT}/include")
    endif()

    find_path(_INFINI_RT_INCLUDE_DIR
        NAMES infini/rt.h
        HINTS ${_infinirt_include_hints}
        PATH_SUFFIXES include
    )

    if(INFINI_RT_LIBRARY)
        set(_INFINI_RT_LIBRARY "${INFINI_RT_LIBRARY}")
    else()
        set(_infinirt_library_hints "")
        if(INFINI_RT_ROOT)
            list(APPEND _infinirt_library_hints
                "${INFINI_RT_ROOT}/lib"
                "${INFINI_RT_ROOT}/lib64")
        endif()
        find_library(_INFINI_RT_LIBRARY
            NAMES infinirt
            HINTS ${_infinirt_library_hints}
        )
    endif()

    if(NOT _INFINI_RT_INCLUDE_DIR OR NOT _INFINI_RT_LIBRARY)
        message(FATAL_ERROR
            "InfiniRT installed headers/library were not found. "
            "Set `INFINI_RT_ROOT` to an installed InfiniRT prefix, or set "
            "`INFINI_RT_INCLUDE_DIRS` and `INFINI_RT_LIBRARY` explicitly. "
            "For development builds from source, set `INFINI_RT_SOURCE_DIR`.")
    endif()

    set(_infinirt_resolved_include_dirs "")
    foreach(_infinirt_include_dir IN LISTS _infinirt_include_hints)
        if(_infinirt_include_dir AND EXISTS "${_infinirt_include_dir}")
            list(APPEND _infinirt_resolved_include_dirs "${_infinirt_include_dir}")
        endif()
    endforeach()
    list(APPEND _infinirt_resolved_include_dirs "${_INFINI_RT_INCLUDE_DIR}")
    list(REMOVE_DUPLICATES _infinirt_resolved_include_dirs)

    add_library(infinirt SHARED IMPORTED GLOBAL)
    set_target_properties(infinirt PROPERTIES
        IMPORTED_LOCATION "${_INFINI_RT_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${_infinirt_resolved_include_dirs}"
    )

    set(INFINI_RT_INCLUDE_DIRS "${_infinirt_resolved_include_dirs}" CACHE STRING "InfiniRT include directories" FORCE)
    set(INFINI_RT_LIBRARY "${_INFINI_RT_LIBRARY}" CACHE FILEPATH "InfiniRT library" FORCE)
    message(STATUS "Using installed InfiniRT: ${_INFINI_RT_LIBRARY}")
endfunction()

_infiniops_configure_infinirt()

add_subdirectory(src)

if(NOT GENERATE_PYTHON_BINDINGS)
    add_subdirectory(examples)
endif()

if(INFINI_OPS_BUILD_DOCS)
    include(FetchContent)
    find_package(Doxygen REQUIRED)

    FetchContent_Declare(
        doxygen_awesome
        URL https://github.com/jothepro/doxygen-awesome-css/archive/refs/tags/v2.4.2.tar.gz
        URL_HASH SHA256=6f527e5733723bfd7243c28f24588dca4714cf977fea6147fdb4ae2050244409
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(doxygen_awesome)

    set(INFINI_OPS_DOXYGEN_OUTPUT_DIRECTORY
        "${CMAKE_BINARY_DIR}/docs/reference")
    set(INFINI_OPS_DOXYGEN_AWESOME_STYLESHEET
        "${doxygen_awesome_SOURCE_DIR}/doxygen-awesome.css")
    set(INFINI_OPS_DOXYGEN_STYLESHEET
        "${CMAKE_CURRENT_SOURCE_DIR}/docs/assets/infini-ops.css")
    set(INFINI_OPS_DOXYGEN_AWESOME_LICENSE
        "${doxygen_awesome_SOURCE_DIR}/LICENSE")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in"
        "${CMAKE_BINARY_DIR}/docs/Doxyfile"
        @ONLY)

    add_custom_target(infiniops_docs
        COMMAND ${CMAKE_COMMAND} -E make_directory
                "${INFINI_OPS_DOXYGEN_OUTPUT_DIRECTORY}"
        COMMAND ${DOXYGEN_EXECUTABLE} "${CMAKE_BINARY_DIR}/docs/Doxyfile"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${INFINI_OPS_DOXYGEN_AWESOME_LICENSE}"
                "${INFINI_OPS_DOXYGEN_OUTPUT_DIRECTORY}/html/doxygen-awesome.LICENSE"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating InfiniOps documentation"
        VERBATIM)
endif()
