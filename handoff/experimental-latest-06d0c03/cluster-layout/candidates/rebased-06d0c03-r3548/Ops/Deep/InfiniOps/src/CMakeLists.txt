add_library(infiniops SHARED operator.h)
add_library(InfiniOps::infiniops ALIAS infiniops)
set_target_properties(infiniops PROPERTIES LINKER_LANGUAGE CXX)

# Compatibility implementations must reference deprecated public classes.
target_compile_options(infiniops PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX,CUDA>:-Wno-deprecated-declarations>"
)

if(INFINI_OPS_ENABLE_HOST_RANGE_PROFILING)
    target_compile_definitions(infiniops PUBLIC
        INFINI_OPS_ENABLE_HOST_RANGE_PROFILING=1)
endif()

if(GENERATE_OPERATOR_CALL_INSTANTIATIONS)
    target_compile_definitions(infiniops PRIVATE
        INFINI_OPS_USE_OPERATOR_CALL_INSTANTIATIONS=1)
endif()

function(_infini_ops_write_if_different path content)
    set(_should_write TRUE)
    if(EXISTS "${path}")
        file(READ "${path}" _existing_content)
        if(_existing_content STREQUAL "${content}")
            set(_should_write FALSE)
        endif()
    endif()

    if(_should_write)
        get_filename_component(_parent_dir "${path}" DIRECTORY)
        file(MAKE_DIRECTORY "${_parent_dir}")
        file(WRITE "${path}" "${content}")
    endif()
endfunction()

function(_infini_ops_prune_stale_cc_files dir)
    cmake_parse_arguments(PARSE_ARGV 1 _ARG "" "" "EXPECTED")

    if(NOT EXISTS "${dir}")
        return()
    endif()

    file(GLOB _existing_cc_files "${dir}/*.cc")
    foreach(_path IN LISTS _existing_cc_files)
        list(FIND _ARG_EXPECTED "${_path}" _index)
        if(_index EQUAL -1)
            file(REMOVE "${_path}")
        endif()
    endforeach()
endfunction()

include(GNUInstallDirs)

find_package(nlohmann_json 3.12.0 CONFIG QUIET)
if(NOT TARGET nlohmann_json::nlohmann_json)
    if(POLICY CMP0135)
        cmake_policy(SET CMP0135 NEW)
    endif()
    include(FetchContent)
    FetchContent_Declare(nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
        URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

file(GLOB BASE_SRCS CONFIGURE_DEPENDS "*.cc")
list(FILTER BASE_SRCS EXCLUDE REGEX ".*tensor\\.cc$")
target_sources(infiniops PRIVATE ${BASE_SRCS})

target_link_libraries(infiniops PUBLIC
    $<BUILD_INTERFACE:infinirt>
    $<INSTALL_INTERFACE:InfiniRT::infinirt>
)
target_link_libraries(infiniops PRIVATE nlohmann_json::nlohmann_json)

# Native SDK providers need explicit Python linkage when Torch/linked providers
# bring torch_python or Python extension DSOs into a standalone SDK consumer.
if(WITH_TORCH)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    target_link_libraries(infiniops PRIVATE -Wl,--no-as-needed Python::Python -Wl,--as-needed)
    if(WITH_HYGON)
        find_library(INFINIOPS_HIP_RUNTIME NAMES amdhip64
            HINTS "${DTK_ROOT}" "${DTK_ROOT}/hip" "$ENV{TORCH_LIB_DIR}"
            PATH_SUFFIXES lib lib64 REQUIRED)
        target_link_libraries(infiniops PRIVATE "${INFINIOPS_HIP_RUNTIME}")
        # DTK compiles generated dispatch units with Clang OpenMP. Their __kmpc
        # entry points are supplied by libomp, not GCC's libgomp.
        find_library(INFINIOPS_DTK_OPENMP NAMES omp
            HINTS "${DTK_ROOT}/llvm" "${DTK_ROOT}" /usr/lib/llvm-14
            PATH_SUFFIXES lib lib64 REQUIRED)
        target_link_libraries(infiniops PRIVATE "${INFINIOPS_DTK_OPENMP}")

    endif()

endif()
if(UNIX AND NOT APPLE)
    target_link_options(infiniops PRIVATE "-Wl,--no-undefined")
endif()


set(INFINI_RT_INCLUDE_FLAGS "")
foreach(_include_dir IN LISTS INFINI_RT_INCLUDE_DIRS)
    list(APPEND INFINI_RT_INCLUDE_FLAGS "-I${_include_dir}")
endforeach()
list(JOIN INFINI_RT_INCLUDE_DIRS ":" INFINI_RT_INCLUDE_DIRS_ENV)

set(DEVICE_LIST "")

if(WITH_CPU)
    set(CPU_PATTERNS
        "native/cpu/*.cc"
        "native/cpu/*.cpp"
    )

    file(GLOB_RECURSE CPU_SOURCES CONFIGURE_DEPENDS ${CPU_PATTERNS})
    target_sources(infiniops PRIVATE ${CPU_SOURCES})

    target_compile_definitions(infiniops PUBLIC WITH_CPU=1)

    find_package(OpenMP REQUIRED COMPONENTS CXX)
    target_link_libraries(infiniops PRIVATE OpenMP::OpenMP_CXX)
    # Generated Call/dispatch translation units use CUDA when NVIDIA is
    # enabled, but also instantiate CPU kernels. OpenMP::OpenMP_CXX only adds
    # compile flags to CXX sources; forward them to NVCC's host compiler too.
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        separate_arguments(_cpu_openmp_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
        foreach(_flag IN LISTS _cpu_openmp_flags)
            target_compile_options(infiniops PRIVATE
                "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${_flag}>")
        endforeach()
    endif()

    list(APPEND DEVICE_LIST "cpu")
endif()

if(WITH_NVIDIA)
    if(NOT TARGET nvidia::cutlass::cutlass)
        find_package(NvidiaCutlass 4.4.1 EXACT CONFIG QUIET)
    endif()

    if(NOT TARGET nvidia::cutlass::cutlass)
        include(FetchContent)
        cmake_policy(PUSH)
        if(POLICY CMP0135)
            cmake_policy(SET CMP0135 NEW)
        endif()
        if(POLICY CMP0169)
            cmake_policy(SET CMP0169 OLD)
        endif()

        FetchContent_Declare(
            cutlass
            URL https://github.com/NVIDIA/cutlass/archive/087c84df83d254b5fb295a7a408f1a1d554085cf.tar.gz
            URL_HASH SHA256=99340f7a8c87f5295b91aecf8295df6693ad954807672c3869a7fb3d0084f709
        )
        # Populate only to avoid CUTLASS 4.4.1's CMake 3.19 requirement and install rules.
        FetchContent_GetProperties(cutlass)
        if(NOT cutlass_POPULATED)
            FetchContent_Populate(cutlass)
        endif()
        cmake_policy(POP)

        if(NOT EXISTS "${cutlass_SOURCE_DIR}/include/cutlass/cutlass.h")
            message(FATAL_ERROR
                "CUTLASS source directory `${cutlass_SOURCE_DIR}` does not contain "
                "`include/cutlass/cutlass.h`.")
        endif()

        add_library(infiniops_cutlass_headers INTERFACE)
        target_include_directories(infiniops_cutlass_headers INTERFACE
            "${cutlass_SOURCE_DIR}/include")
        add_library(nvidia::cutlass::cutlass ALIAS infiniops_cutlass_headers)
    endif()

    set(NVIDIA_PATTERNS
        "native/cuda/*.cc"
        "native/cuda/*.cpp"
        "native/cuda/*.cu"
        "native/cuda/nvidia/*.cc"
        "native/cuda/nvidia/*.cpp"
        "native/cuda/nvidia/*.cu"
    )

    file(GLOB_RECURSE NVIDIA_SOURCES CONFIGURE_DEPENDS ${NVIDIA_PATTERNS})
    # Optional pinned kernels have their own include closure and CUDA TUs.
    list(FILTER NVIDIA_SOURCES EXCLUDE REGEX "/ops/flash_attn_with_kvcache/")
    list(FILTER NVIDIA_SOURCES EXCLUDE REGEX "/ops/flash_attn_varlen_func/")
    list(FILTER NVIDIA_SOURCES EXCLUDE REGEX "/ops/flash_common\\.cc$")

    # The CUTLASS INT8 Tensor Core kernel needs SM75 device compilation. Scope
    # the extra code objects to this provider source so other CUDA operators
    # retain the caller/compiler architecture configuration unchanged.
    set(_cutlass_scaled_mm_source
        "${CMAKE_CURRENT_SOURCE_DIR}/native/cuda/nvidia/ops/cutlass_scaled_mm/cutlass.cu")
    set_source_files_properties("${_cutlass_scaled_mm_source}" PROPERTIES
        COMPILE_OPTIONS
            "--generate-code=arch=compute_75,code=sm_75;--generate-code=arch=compute_75,code=compute_75")

    list(FILTER NVIDIA_SOURCES EXCLUDE REGEX "/ops/w4_marlin/")
    enable_language(CUDA)
    if(BUILD_NVIDIA_W4_MARLIN)
        include("${PROJECT_SOURCE_DIR}/cmake/NativeW4Marlin.cmake")
    endif()

    if(BUILD_NVIDIA_FLASH_INFERENCE)
        include("${PROJECT_SOURCE_DIR}/cmake/NativeFlashAttention.cmake")
        infiniops_add_native_flash(infiniops_native_flash)
        target_link_libraries(infiniops PRIVATE infiniops_native_flash)
    endif()

    target_compile_definitions(infiniops PUBLIC WITH_NVIDIA=1)
    target_sources(infiniops PRIVATE ${NVIDIA_SOURCES})
    target_link_libraries(infiniops PRIVATE nvidia::cutlass::cutlass)

    target_link_libraries(infiniops PUBLIC CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cuda_driver)

    list(APPEND DEVICE_LIST "nvidia")
    set_target_properties(infiniops PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
endif()

if(WITH_NINETOOTHED)
    find_package(Python COMPONENTS Interpreter REQUIRED)

    if(NINETOOTHED_PYTHON_EXECUTABLE)
        set(_ninetoothed_python "${NINETOOTHED_PYTHON_EXECUTABLE}")
    elseif(_TORCH_PYTHON)
        set(_ninetoothed_python "${_TORCH_PYTHON}")
    else()
        set(_ninetoothed_python "${Python_EXECUTABLE}")
    endif()
    message(STATUS "NineToothed codegen Python: ${_ninetoothed_python}")

    set(_ninetoothed_output_dir "${CMAKE_CURRENT_BINARY_DIR}/ninetoothed")
    set(_ninetoothed_generator_args
        "${PROJECT_SOURCE_DIR}/scripts/generate_ninetoothed_ops.py"
        --output-dir "${_ninetoothed_output_dir}")

    execute_process(
        COMMAND "${_ninetoothed_python}" ${_ninetoothed_generator_args}
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        RESULT_VARIABLE _ninetoothed_generation_result
    )

    if(NOT _ninetoothed_generation_result EQUAL 0)
        message(FATAL_ERROR "Generating NineToothed operator sources failed with `${_ninetoothed_python}`. Set `NINETOOTHED_PYTHON_EXECUTABLE` to a Python with `ninetoothed`, `ntops`, `triton`, `sympy`, and CUDA dependencies installed.")
    endif()

    include("${_ninetoothed_output_dir}/manifest.cmake")
    target_include_directories(infiniops PRIVATE
        ${INFINI_OPS_NINETOOTHED_INCLUDE_DIRS})
    target_sources(infiniops PRIVATE ${INFINI_OPS_NINETOOTHED_SOURCES})
endif()

if(WITH_TRITON OR GENERATE_PYTHON_BINDINGS)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)

    if(NOT pybind11_DIR)
        execute_process(
            COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
            OUTPUT_VARIABLE _pybind11_cmake_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _pybind11_result
        )

        if(_pybind11_result EQUAL 0)
            set(pybind11_DIR "${_pybind11_cmake_dir}" CACHE PATH "pybind11 CMake directory")
        endif()
    endif()

    find_package(pybind11 CONFIG REQUIRED)
endif()

if(WITH_TRITON)
    target_link_libraries(infiniops PRIVATE
        Python::Python
        pybind11::headers)
    target_sources(infiniops PRIVATE
        triton/jit/cache.cc
        triton/jit/compiler.cc)
endif()

if(WITH_ILUVATAR)
    set(ILUVATAR_PATTERNS
        "native/cuda/*.cc"
        "native/cuda/*.cpp"
        "native/cuda/*.cu"
        "native/cuda/iluvatar/*.cc"
        "native/cuda/iluvatar/*.cpp"
        "native/cuda/iluvatar/*.cu"
    )

    file(GLOB_RECURSE ILUVATAR_SOURCES CONFIGURE_DEPENDS ${ILUVATAR_PATTERNS})
    list(FILTER ILUVATAR_SOURCES EXCLUDE REGEX
        ".*/native/cuda/(hygon|mars|metax|moore|nvidia|thead)/.*")

    target_compile_definitions(infiniops PUBLIC WITH_ILUVATAR=1)
    target_sources(infiniops PRIVATE ${ILUVATAR_SOURCES})

    find_package(CUDAToolkit REQUIRED)
    # CoreX does not provide FindCUDAToolkit's CUDA::cuda_driver target. The
    # Iluvatar backend uses the runtime and BLAS APIs directly.
    target_link_libraries(infiniops PUBLIC CUDA::cudart CUDA::cublas)

    list(APPEND DEVICE_LIST "iluvatar")
endif()

if(WITH_HYGON)
    set(HYGON_PATTERNS
        "native/cuda/*.cc"
        "native/cuda/*.cpp"
        "native/cuda/*.cu"
        "native/cuda/hygon/*.cc"
        "native/cuda/hygon/*.cpp"
        "native/cuda/hygon/*.cu"
    )

    file(GLOB_RECURSE HYGON_SOURCES CONFIGURE_DEPENDS ${HYGON_PATTERNS})
    list(FILTER HYGON_SOURCES EXCLUDE REGEX
        ".*/native/cuda/(iluvatar|metax|moore|nvidia)/.*")

    enable_language(CUDA)

    target_compile_definitions(infiniops PUBLIC WITH_HYGON=1)
    target_sources(infiniops PRIVATE ${HYGON_SOURCES})

    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(infiniops PUBLIC CUDA::cudart CUDA::cublas CUDA::cuda_driver)

    set_target_properties(infiniops PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )

    list(APPEND DEVICE_LIST "hygon")
endif()

if(WITH_THEAD)
    set(THEAD_PATTERNS
        "native/cuda/thead/*.cc"
        "native/cuda/thead/*.cpp"
        "native/cuda/thead/*.cu"
    )

    file(GLOB_RECURSE THEAD_SOURCES CONFIGURE_DEPENDS ${THEAD_PATTERNS})

    enable_language(CUDA)

    target_compile_definitions(infiniops PUBLIC WITH_THEAD=1)
    target_sources(infiniops PRIVATE ${THEAD_SOURCES})

    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(infiniops PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::cuda_driver)

    set_target_properties(infiniops PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )

    list(APPEND DEVICE_LIST "thead")
endif()

if(WITH_METAX)
    set(METAX_PATTERNS
        "native/cuda/*.cc"
        "native/cuda/*.cpp"
        "native/cuda/metax/*.cc"
        "native/cuda/metax/*.maca"
    )

    file(GLOB_RECURSE METAX_SOURCES CONFIGURE_DEPENDS ${METAX_PATTERNS})
    list(FILTER METAX_SOURCES EXCLUDE REGEX
        ".*/native/cuda/(hygon|iluvatar|mars|moore|nvidia|thead)/.*")

    set_source_files_properties(${METAX_SOURCES} PROPERTIES LANGUAGE CXX)

    target_compile_definitions(infiniops PRIVATE WITH_METAX=1)
    target_compile_options(infiniops PRIVATE "-x" "maca")
    target_sources(infiniops PRIVATE ${METAX_SOURCES})

    target_include_directories(infiniops PUBLIC "${MACA_PATH}/include")
    target_link_libraries(infiniops PUBLIC 
        ${MACA_RUNTIME_LIB} 
        ${MACA_DNN_LIB} 
        ${MACA_BLAS_LIB}
    )

    list(APPEND DEVICE_LIST "metax")
endif()

if(WITH_MARS)
    set(MARS_PATTERNS
        "native/cuda/*.cc"
        "native/cuda/*.cpp"
        "native/cuda/mars/*.cc"
        "native/cuda/mars/*.cu"
    )

    file(GLOB_RECURSE MARS_SOURCES CONFIGURE_DEPENDS ${MARS_PATTERNS})
    list(FILTER MARS_SOURCES EXCLUDE REGEX
        ".*/native/cuda/(hygon|iluvatar|metax|moore|nvidia|thead)/.*")

    set_source_files_properties(${MARS_SOURCES} PROPERTIES LANGUAGE CXX)

    target_compile_definitions(infiniops PRIVATE WITH_MARS=1)
    target_sources(infiniops PRIVATE ${MARS_SOURCES})

    target_include_directories(infiniops PUBLIC
        "${HPCC_PATH}/include"
        "${HPCC_PATH}/tools/cu-bridge/include")
    target_link_libraries(infiniops PUBLIC
        ${HPCC_RUNTIME_LIB}
        ${HPCC_DNN_LIB}
        ${HPCC_BLAS_LIB}
        ${HPCC_OMP_LIB}
    )
    set_property(TARGET infiniops APPEND PROPERTY
        INSTALL_RPATH "${HPCC_PATH}/htgpu_llvm/lib")
    set_property(TARGET infiniops APPEND PROPERTY
        BUILD_RPATH "${HPCC_PATH}/htgpu_llvm/lib")

    list(APPEND DEVICE_LIST "mars")
endif()

if(WITH_MOORE)
    set(MOORE_PATTERNS
        "native/cuda/*.cc"
        "native/cuda/*.cpp"
        "native/cuda/moore/*.cc"
        "native/cuda/moore/*.cpp"
        "native/cuda/moore/*.mu"
    )

    file(GLOB_RECURSE MOORE_SOURCES CONFIGURE_DEPENDS ${MOORE_PATTERNS})
    list(FILTER MOORE_SOURCES EXCLUDE REGEX
        ".*/native/cuda/(hygon|iluvatar|mars|metax|nvidia|thead)/.*")

    set_source_files_properties(${MOORE_SOURCES} PROPERTIES LANGUAGE CXX)

    target_compile_definitions(infiniops PRIVATE WITH_MOORE=1)
    target_compile_options(infiniops PRIVATE "-x" "musa")
    target_sources(infiniops PRIVATE ${MOORE_SOURCES})

    target_include_directories(infiniops PUBLIC "${MUSA_ROOT}/include")
    target_link_libraries(infiniops PUBLIC ${MUSA_LIB} ${MUSART_LIB} ${MUBLAS_LIB} ${MUSA_OPENMP_LIB})

    list(APPEND DEVICE_LIST "moore")
endif()

if(WITH_CAMBRICON)
    file(GLOB_RECURSE CAMBRICON_MLU_SOURCES CONFIGURE_DEPENDS "native/cambricon/ops/*/*.mlu")
    find_program(CNCC_COMPILER cncc HINTS "${NEUWARE_HOME}/bin" "$ENV{NEUWARE_HOME}/bin" /usr/local/neuware/bin)
    if(CNCC_COMPILER)
        message(STATUS "Found cncc: ${CNCC_COMPILER}")
        separate_arguments(INFINI_MLU_HOST_ABI_FLAGS NATIVE_COMMAND "${CMAKE_CXX_FLAGS}")
        set(MLU_COMPILE_OPTS
            -c --bang-mlu-arch=mtp_592 -O3 -fPIC -Wall -Werror -std=c++17 -pthread
            -I${CMAKE_CURRENT_SOURCE_DIR} -I${NEUWARE_HOME}/include
            ${INFINI_RT_INCLUDE_FLAGS}
            ${INFINI_MLU_HOST_ABI_FLAGS}
            -idirafter /usr/local/neuware/lib/clang/11.1.0/include
        )
        function(compile_mlu_file src_file)
            get_filename_component(name ${src_file} NAME_WE)
            get_filename_component(path ${src_file} DIRECTORY)
            set(out_file "${CMAKE_CURRENT_BINARY_DIR}/${path}/${name}.o")
            file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${path}")
            add_custom_command(OUTPUT ${out_file}
                COMMAND ${CNCC_COMPILER} ${MLU_COMPILE_OPTS} -c ${src_file} -o ${out_file}
                DEPENDS ${src_file}
                COMMENT "Building MLU kernel: ${src_file}"
            )
            set_property(DIRECTORY APPEND PROPERTY CAMBRICON_OBJECTS ${out_file})
        endfunction()
        foreach(src ${CAMBRICON_MLU_SOURCES})
            compile_mlu_file(${src})
        endforeach()
        get_directory_property(CAMBRICON_OBJECT_FILES CAMBRICON_OBJECTS)
        if(CAMBRICON_OBJECT_FILES)
            target_sources(infiniops PRIVATE ${CAMBRICON_OBJECT_FILES})
        endif()
    else()
        message(WARNING "cncc compiler not found. MLU kernels will not be compiled.")
    endif()
    target_compile_definitions(infiniops PRIVATE WITH_CAMBRICON=1)

    target_include_directories(infiniops PUBLIC "${NEUWARE_HOME}/include")
    target_link_libraries(infiniops PUBLIC ${CAMBRICON_RUNTIME_LIB} ${CAMBRICON_CNNL_LIB} ${CAMBRICON_CNNL_EXTRA_LIB} ${CAMBRICON_PAPI_LIB})

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(infiniops PUBLIC 
            "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-idirafter /usr/local/neuware/lib/clang/11.1.0/include>"
        )
        set(_cambricon_host_bang_compat_option
            "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-include ${CMAKE_CURRENT_SOURCE_DIR}/native/cambricon/host_bang_compat.h>")
        target_compile_options(infiniops PRIVATE
            "${_cambricon_host_bang_compat_option}")
    endif()

    list(APPEND DEVICE_LIST "cambricon")
endif()

if(WITH_ASCEND)
    # ASCEND_HOME is set by the top-level CMakeLists.txt.
    file(GLOB_RECURSE ASCEND_SOURCES CONFIGURE_DEPENDS
        "native/ascend/*.cc"
        "native/ascend/*.cpp"
    )
    # Exclude `kernel_impl.cpp` — `AscendC` device code, not compiled by the host C++ compiler.
    list(FILTER ASCEND_SOURCES EXCLUDE REGEX ".*kernel_impl\\.cpp$")
    # Exclude custom/ — standalone PyTorch extension, built separately.
    list(FILTER ASCEND_SOURCES EXCLUDE REGEX ".*/custom/.*")

    target_compile_definitions(infiniops PUBLIC WITH_ASCEND=1)
    target_sources(infiniops PRIVATE ${ASCEND_SOURCES})

    # Resolve the driver lib dir two levels above the toolkit root.
    get_filename_component(ASCEND_ROOT "${ASCEND_HOME}/../.." ABSOLUTE)

    # Prefer the real driver HAL; fall back to the toolkit stub for build-only
    # environments (e.g., Docker CI images without hardware drivers installed).
    # CANN <= 8.0: stub at runtime/lib64/stub/; CANN >= 8.5: devlib/<arch>-linux/devlib/.
    set(ASCEND_HAL_REAL   "${ASCEND_ROOT}/driver/lib64/driver/libascend_hal.so")
    set(ASCEND_HAL_STUB   "${ASCEND_HOME}/runtime/lib64/stub/libascend_hal.so")
    set(ASCEND_HAL_DEVLIB "${ASCEND_HOME}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib/libascend_hal.so")
    if(EXISTS "${ASCEND_HAL_REAL}")
        set(ASCEND_HAL_LIB "${ASCEND_HAL_REAL}")
    elseif(EXISTS "${ASCEND_HAL_STUB}")
        set(ASCEND_HAL_LIB "${ASCEND_HAL_STUB}")
        message(STATUS "ascend_hal: driver not found, using stub for linking")
    elseif(EXISTS "${ASCEND_HAL_DEVLIB}")
        set(ASCEND_HAL_LIB "${ASCEND_HAL_DEVLIB}")
        message(STATUS "ascend_hal: driver not found, using devlib for linking")
    else()
        message(FATAL_ERROR "libascend_hal.so not found (tried ${ASCEND_HAL_REAL}, ${ASCEND_HAL_STUB}, and ${ASCEND_HAL_DEVLIB})")
    endif()

    target_include_directories(infiniops PUBLIC
        "${ASCEND_HOME}/include"
        "${ASCEND_HOME}/include/aclnn"
        "${ASCEND_HOME}/include/aclnnop")
    target_link_libraries(infiniops PUBLIC
        "${ASCEND_HOME}/lib64/libascendcl.so"
        "${ASCEND_HOME}/lib64/libnnopbase.so"
        "${ASCEND_HOME}/lib64/libopapi.so"
        "${ASCEND_HAL_LIB}")

    # ATB (Ascend Transformer Boost) — provides fused operators like
    # `PagedAttention` and `ReshapeAndCache` that are graph-capture safe.
    set(ATB_HOME_DIR "$ENV{ATB_HOME_PATH}")
    if(NOT ATB_HOME_DIR)
        # Default search path under CANN nnal directory.
        file(GLOB ATB_SEARCH_DIRS "/usr/local/Ascend/nnal/atb/*/atb/cxx_abi_1")
        if(ATB_SEARCH_DIRS)
            list(SORT ATB_SEARCH_DIRS ORDER DESCENDING)
            list(GET ATB_SEARCH_DIRS 0 ATB_HOME_DIR)
        endif()
    endif()

    if(ATB_HOME_DIR AND EXISTS "${ATB_HOME_DIR}/include/atb/operation.h")
        message(STATUS "ATB found: ${ATB_HOME_DIR}")
        target_compile_definitions(infiniops PUBLIC INFINI_HAS_ATB=1)
        target_include_directories(infiniops PUBLIC "${ATB_HOME_DIR}/include")
        target_link_libraries(infiniops PUBLIC "${ATB_HOME_DIR}/lib/libatb.so")
    else()
        message(STATUS "ATB not found — ATB-based operators disabled")
    endif()

    list(APPEND DEVICE_LIST "ascend")

    # Custom `AscendC` kernels (PyTorch extension, requires `torch_npu`).
    if(BUILD_ASCEND_CUSTOM)
        # In-tree `ascendc_library()` trips the `CANN` `extract_host_stub.py`
        # path-handling bug under `scikit-build-core`'s temp-dir builds
        # (`KeyError` on `/./workspace/...` paths in `$<TARGET_OBJECTS>`).
        # Work around it by driving the standalone `src/native/ascend/custom/build.sh`;
        # that script invokes a separate `cmake` with
        # `src/native/ascend/custom/` as its `SOURCE_DIR`, avoiding the buggy
        # path shape.  The produced `.a` is imported and linked into
        # `ops` with `--whole-archive`.
        set(_custom_build_dir "${CMAKE_SOURCE_DIR}/build/build_ascend_custom")
        set(_custom_lib "${_custom_build_dir}/lib/libno_workspace_kernel.a")
        set(_custom_source_dir "${CMAKE_CURRENT_BINARY_DIR}/ascend_custom_source")

        if(NOT DEFINED SOC_VERSION OR "${SOC_VERSION}" STREQUAL "")
            include(${CMAKE_CURRENT_SOURCE_DIR}/native/ascend/custom/cmake/detect_soc.cmake)
            infiniops_detect_soc(SOC_VERSION)
        endif()

        # Drive `build.sh` as a build-phase target with explicit source
        # dependencies so that editing any `op_host/` or `op_kernel/`
        # source re-triggers the build (plain `execute_process` at
        # configure time would only gate on file existence and leave
        # stale `.a` files in place).
        file(GLOB_RECURSE _custom_srcs CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/native/ascend/custom/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/native/ascend/custom/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/native/ascend/custom/*.cmake"
            "${CMAKE_CURRENT_SOURCE_DIR}/native/ascend/custom/CMakeLists.txt"
            "${CMAKE_CURRENT_SOURCE_DIR}/native/ascend/custom/build.sh")

        # Scrub env inherited from the outer `scikit-build-core` invocation
        # before handing control to `build.sh`: `CMAKE_GENERATOR` /
        # `CMAKE_EXPORT_COMPILE_COMMANDS` leaking into the inner `cmake`
        # change the path format passed to `ninja`'s `_host_cpp` rule and
        # re-trigger the `CANN` `extract_host_stub.py` `KeyError`
        # (`/./workspace/...`) that standalone `build.sh` avoids.
        #
        # `pip install` MUST be invoked with `--no-build-isolation` on
        # Ascend; otherwise pip's build-isolation overlay shadows system
        # `torch` (via `PYTHONPATH`) and the inner `cmake`'s
        # `import torch` in `config_envs.cmake` fails with
        # `ModuleNotFoundError`.
        add_custom_command(
            OUTPUT ${_custom_lib}
            COMMAND ${CMAKE_COMMAND} -E rm -f "${_custom_source_dir}"
            COMMAND ${CMAKE_COMMAND} -E create_symlink
                    "${CMAKE_CURRENT_SOURCE_DIR}/native/ascend/custom"
                    "${_custom_source_dir}"
            COMMAND ${CMAKE_COMMAND} -E env
                    --unset=CMAKE_GENERATOR
                    --unset=CMAKE_EXPORT_COMPILE_COMMANDS
                    --unset=CMAKE_BUILD_PARALLEL_LEVEL
                    "BUILD_DIR=${_custom_build_dir}"
                    "MAIN_SRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
                    bash ${_custom_source_dir}/build.sh ${SOC_VERSION}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/native/ascend/custom
            DEPENDS ${_custom_srcs}
            COMMENT "Building custom AscendC kernels (SOC_VERSION=${SOC_VERSION})"
            VERBATIM)

        add_custom_target(no_workspace_kernel_build ALL DEPENDS ${_custom_lib})

        add_library(no_workspace_kernel STATIC IMPORTED GLOBAL)
        set_target_properties(no_workspace_kernel PROPERTIES
            IMPORTED_LOCATION "${_custom_lib}")
        add_dependencies(no_workspace_kernel no_workspace_kernel_build)

        # Link the compiled `AscendC` kernel objects into `infiniops` so that
        # custom kernel implementations (e.g. `RmsNorm` index 1) can call
        # them via the generated launch functions.
        target_compile_definitions(infiniops PUBLIC INFINI_HAS_CUSTOM_KERNELS=1)
        target_include_directories(infiniops PRIVATE
            "${_custom_build_dir}/include/no_workspace_kernel")
        target_link_libraries(infiniops PRIVATE
            -Wl,--whole-archive $<TARGET_FILE:no_workspace_kernel> -Wl,--no-whole-archive)
        add_dependencies(infiniops no_workspace_kernel_build)
    endif()
endif()

set(INFINI_OPS_OPS "" CACHE STRING
    "Operator allowlist or path to an ops.json implementation selection")
set(INFINI_OPS_SMOKE_BUILD OFF CACHE BOOL
    "Build only the smoke-test operator subset")
set(_infini_ops_smoke_ops
    add mul relu cast cat gemm matmul linear rms_norm swiglu silu_and_mul
    causal_softmax causal_softmax_infinilm abs clamp exp)
set(_infini_ops_smoke_torch_ops abs clamp exp relu)

if(WITH_NVIDIA)
    list(APPEND _infini_ops_smoke_ops cutlass_scaled_mm moe_sum)
endif()

if(((WITH_NVIDIA OR WITH_MOORE) AND WITH_TORCH) OR WITH_ILUVATAR)
    list(APPEND _infini_ops_smoke_ops flash_attn_varlen_func)
endif()

if(WITH_ILUVATAR)
    list(APPEND _infini_ops_smoke_ops flash_attn_with_kvcache)
endif()

if(INFINI_OPS_SMOKE_BUILD)
    if(NOT INFINI_OPS_OPS)
        set(INFINI_OPS_OPS "${_infini_ops_smoke_ops}" CACHE STRING
            "Semicolon- or comma-separated operator allowlist for generated wrappers and bindings" FORCE)
    endif()
    if(NOT INFINI_OPS_TORCH_OPS)
        set(INFINI_OPS_TORCH_OPS "${_infini_ops_smoke_torch_ops}" CACHE STRING
            "Semicolon-separated PyTorch op allowlist for generated torch wrappers" FORCE)
    endif()
endif()

set(_infini_ops_ops_config "")
if(INFINI_OPS_OPS MATCHES "\\.json$")
    get_filename_component(_infini_ops_ops_config "${INFINI_OPS_OPS}"
        ABSOLUTE BASE_DIR "${PROJECT_SOURCE_DIR}")
    if(NOT EXISTS "${_infini_ops_ops_config}")
        message(FATAL_ERROR
            "Operator selection `${_infini_ops_ops_config}` does not exist.")
    endif()
elseif(NOT INFINI_OPS_OPS AND EXISTS "${PROJECT_SOURCE_DIR}/ops.json")
    set(_infini_ops_ops_config "${PROJECT_SOURCE_DIR}/ops.json")
endif()

if(_infini_ops_ops_config)
    file(GLOB_RECURSE _infini_ops_implementation_headers CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/src/native/*.h"
        "${PROJECT_SOURCE_DIR}/src/torch/*.h"
        "${PROJECT_SOURCE_DIR}/src/ninetoothed/*.h"
        "${PROJECT_SOURCE_DIR}/src/linked/*.h"
        "${PROJECT_SOURCE_DIR}/src/triton/*.h")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${_infini_ops_ops_config}"
        "${PROJECT_SOURCE_DIR}/scripts/ops_config.py"
        ${_infini_ops_implementation_headers})
    message(STATUS "Operator selection: ${_infini_ops_ops_config}")
endif()

if(INFINI_OPS_OPS AND NOT INFINI_OPS_OPS MATCHES "\\.json$")
    string(REPLACE "," ";" _infini_ops_op_allowlist "${INFINI_OPS_OPS}")
    message(STATUS "Wrapper op allowlist: ${_infini_ops_op_allowlist}")
endif()

set(INFINI_OPS_LINKED_SOURCES "")
set(INFINI_OPS_LINKED_TORCH_SOURCES "")
set(INFINI_OPS_LINKED_TVM_FFI_SOURCES "")
set(INFINI_OPS_LINKED_LIBRARIES "")
set(INFINI_OPS_LINKED_FORCE_LOAD_LIBRARIES "")
set(INFINI_OPS_LINKED_RUNTIME_DIRS "")
set(INFINI_OPS_LINKED_INCLUDE_DIRS "")
set(INFINI_OPS_LINKED_TRANSPORTS "")
set(_infini_ops_linked_uses_torch FALSE)
set(_infini_ops_linked_uses_tvm_ffi FALSE)

if(WITH_LINKED)
    find_package(Python COMPONENTS Interpreter REQUIRED)
    if(_TORCH_PYTHON)
        set(_linked_python "${_TORCH_PYTHON}")
    else()
        set(_linked_python "${Python_EXECUTABLE}")
    endif()
    if(NOT DEVICE_LIST)
        message(FATAL_ERROR
            "`WITH_LINKED=ON` requires an enabled device backend.")
    endif()

    file(GLOB_RECURSE _linked_resolution_inputs CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/src/linked/*.cc"
        "${PROJECT_SOURCE_DIR}/src/linked/*.cu"
        "${PROJECT_SOURCE_DIR}/src/linked/*.h"
        "${PROJECT_SOURCE_DIR}/src/linked/*.yaml")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/scripts/resolve_linked_ops.py"
        ${_linked_resolution_inputs})

    if(CMAKE_NM AND EXISTS "${CMAKE_NM}")
        set(_linked_nm "${CMAKE_NM}")
    else()
        find_program(_linked_nm NAMES llvm-nm nm)
    endif()
    if(CMAKE_READELF AND EXISTS "${CMAKE_READELF}")
        set(_linked_readelf "${CMAKE_READELF}")
    else()
        find_program(_linked_readelf NAMES llvm-readelf readelf)
    endif()
    find_program(_linked_cxxfilt NAMES llvm-cxxfilt c++filt)
    if(NOT _linked_nm OR NOT _linked_readelf OR NOT _linked_cxxfilt)
        message(FATAL_ERROR
            "`WITH_LINKED=ON` requires compatible `nm`, `readelf`, and "
            "`c++filt` tools.")
    endif()

    set(_linked_resolver_args
        "${PROJECT_SOURCE_DIR}/scripts/resolve_linked_ops.py"
        --devices ${DEVICE_LIST}
        --nm "${_linked_nm}"
        --readelf "${_linked_readelf}"
        --cxxfilt "${_linked_cxxfilt}")
    if(_infini_ops_ops_config)
        list(APPEND _linked_resolver_args
            --ops-config "${_infini_ops_ops_config}")
    endif()
    if(_infini_ops_op_allowlist)
        list(APPEND _linked_resolver_args --ops ${_infini_ops_op_allowlist})
    endif()

    execute_process(
        COMMAND "${_linked_python}" ${_linked_resolver_args}
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        RESULT_VARIABLE _linked_resolution_result
    )
    if(NOT _linked_resolution_result EQUAL 0)
        message(FATAL_ERROR "Resolving linked operator libraries failed.")
    endif()

    include("${PROJECT_SOURCE_DIR}/generated/linked/manifest.cmake")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        ${INFINI_OPS_LINKED_LIBRARIES})

    foreach(_linked_transport IN LISTS INFINI_OPS_LINKED_TRANSPORTS)
        if(_linked_transport STREQUAL "torch")
            set(_infini_ops_linked_uses_torch TRUE)
        elseif(_linked_transport STREQUAL "tvm_ffi")
            set(_infini_ops_linked_uses_tvm_ffi TRUE)
        else()
            message(FATAL_ERROR
                "Unsupported linked operator transport `${_linked_transport}`.")
        endif()
    endforeach()

    if(_infini_ops_linked_uses_torch AND NOT WITH_TORCH)
        infiniops_find_torch()
    endif()

    list(REMOVE_DUPLICATES INFINI_OPS_LINKED_LIBRARIES)
    set(_linked_library_basenames)
    set(_linked_library_index 0)
    foreach(_linked_library IN LISTS INFINI_OPS_LINKED_LIBRARIES)
        get_filename_component(_linked_library_dir
            "${_linked_library}" DIRECTORY)
        get_filename_component(_linked_library_name
            "${_linked_library}" NAME)
        list(FIND _linked_library_basenames "${_linked_library_name}"
            _linked_duplicate_basename)
        if(NOT _linked_duplicate_basename EQUAL -1)
            message(FATAL_ERROR
                "Linked operator libraries contain duplicate basename "
                "`${_linked_library_name}` from different directories.")
        endif()
        list(APPEND _linked_library_basenames "${_linked_library_name}")

        set(_linked_library_target
            "infini_ops_linked_library_${_linked_library_index}")
        list(FIND INFINI_OPS_LINKED_FORCE_LOAD_LIBRARIES
            "${_linked_library}" _linked_force_load)
        if(_linked_force_load EQUAL -1)
            set(_linked_library_link_items "-l:${_linked_library_name}")
        else()
            set(_linked_library_link_items
                "-Wl,--push-state,--no-as-needed"
                "-l:${_linked_library_name}"
                "-Wl,--pop-state")
        endif()
        add_library("${_linked_library_target}" INTERFACE IMPORTED)
        set_target_properties("${_linked_library_target}" PROPERTIES
            INTERFACE_LINK_DIRECTORIES "${_linked_library_dir}"
            INTERFACE_LINK_LIBRARIES "${_linked_library_link_items}")
        # Python extension DSOs generally have no SONAME. Link the exact
        # resolved basename while keeping the install location configurable.
        target_link_libraries(infiniops PRIVATE "${_linked_library_target}")
        set_property(TARGET infiniops APPEND PROPERTY LINK_DEPENDS
            "${_linked_library}")
        math(EXPR _linked_library_index "${_linked_library_index} + 1")
    endforeach()

    foreach(_linked_runtime_dir IN LISTS INFINI_OPS_LINKED_RUNTIME_DIRS)
        set_property(TARGET infiniops APPEND PROPERTY BUILD_RPATH
            "${_linked_runtime_dir}")
    endforeach()

    list(LENGTH INFINI_OPS_LINKED_SOURCES _linked_source_count)
    message(STATUS
        "Resolved ${_linked_source_count} linked operator source(s).")
endif()

set(TORCH_SOURCES "")
if(WITH_TORCH)
    # Auto-generate ATen-backed operator wrappers from `scripts/torch_ops.yaml`.
    # The script writes into `${PROJECT_SOURCE_DIR}/generated/` (gitignored),
    # which we then glob below alongside any hand-written torch sources.
    find_package(Python COMPONENTS Interpreter REQUIRED)

    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/scripts/generate_torch_ops.py"
        "${PROJECT_SOURCE_DIR}/scripts/torch_ops.yaml")

    # Pin codegen to the locally installed torch version so vendor
    # forks (Cambricon's `torch_mlu` 2.1.0, etc.) get a schema whose
    # `at::<op>_out` overloads match the headers they ship.  Without
    # this, the codegen targets v2.4.0 and the build fails on older
    # forks with no-known-conversion errors (e.g. `at::all_out`'s
    # `int64_t dim` vs `OptionalIntArrayRef dim`).
    execute_process(
        COMMAND ${_TORCH_PYTHON} -c
            "import torch; print('v' + torch.__version__.split('+')[0])"
        OUTPUT_VARIABLE _torch_version_tag
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _torch_version_result
    )
    if(NOT _torch_version_result EQUAL 0 OR NOT _torch_version_tag)
        set(_torch_version_tag "v2.4.0")
    endif()
    message(STATUS "Codegen schema: PyTorch ${_torch_version_tag}")

    set(INFINI_OPS_TORCH_OPS "" CACHE STRING
        "Semicolon-separated PyTorch op allowlist for generated torch wrappers")
    set(_torch_codegen_args
        ${PROJECT_SOURCE_DIR}/scripts/generate_torch_ops.py
        --pytorch-version ${_torch_version_tag})
    if(_infini_ops_ops_config)
        list(APPEND _torch_codegen_args
            --ops-config "${_infini_ops_ops_config}")
    endif()
    if(INFINI_OPS_TORCH_OPS)
        string(REPLACE "," ";" _torch_op_allowlist "${INFINI_OPS_TORCH_OPS}")
        list(APPEND _torch_codegen_args --ops ${_torch_op_allowlist})
        message(STATUS "Codegen torch op allowlist: ${_torch_op_allowlist}")
    endif()

    execute_process(
        COMMAND ${Python_EXECUTABLE} ${_torch_codegen_args}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        RESULT_VARIABLE _torch_ops_result
    )
    if(NOT _torch_ops_result EQUAL 0)
        message(FATAL_ERROR "Generating torch op wrappers - failed")
    endif()
    message(STATUS "Generating torch op wrappers - done")

    file(GLOB_RECURSE TORCH_SOURCES CONFIGURE_DEPENDS
        "torch/ops/*.cc" "torch/ops/*.cpp"
        "${PROJECT_SOURCE_DIR}/generated/torch/*.cc"
        "${PROJECT_SOURCE_DIR}/generated/torch/*.cpp"
    )

    foreach(_device IN LISTS DEVICE_LIST)
        file(GLOB_RECURSE _torch_device_sources CONFIGURE_DEPENDS
            "torch/${_device}/*.cc" "torch/${_device}/*.cpp")
        list(APPEND TORCH_SOURCES ${_torch_device_sources})
    endforeach()
endif()

if(_infini_ops_linked_uses_torch)
    list(APPEND TORCH_SOURCES ${INFINI_OPS_LINKED_TORCH_SOURCES})
endif()


if(_infini_ops_linked_uses_tvm_ffi)
    target_sources(infiniops PRIVATE ${INFINI_OPS_LINKED_TVM_FFI_SOURCES})
endif()

if(INFINI_OPS_LINKED_INCLUDE_DIRS)
    target_include_directories(infiniops PRIVATE
        ${INFINI_OPS_LINKED_INCLUDE_DIRS})
endif()

if(WITH_ASCEND AND TORCH_SOURCES)
    execute_process(
        COMMAND "${_TORCH_PYTHON}" -c
            "import pathlib, torch_npu; root = pathlib.Path(torch_npu.__file__).resolve().parent; print(root / 'include', root / 'lib', torch_npu.__version__, sep=';')"
        OUTPUT_VARIABLE _torch_npu_paths
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _torch_npu_paths_result
    )
    if(NOT _torch_npu_paths_result EQUAL 0 OR NOT _torch_npu_paths)
        message(FATAL_ERROR
            "The Ascend PyTorch backend requires an installed `torch_npu` package.")
    endif()
    list(GET _torch_npu_paths 0 _torch_npu_include_dir)
    list(GET _torch_npu_paths 1 _torch_npu_library_dir)
    list(GET _torch_npu_paths 2 _torch_npu_version)

    set(_torch_npu_stream_header
        "${_torch_npu_include_dir}/torch_npu/csrc/core/npu/NPUStream.h")
    set(_torch_npu_guard_header
        "${_torch_npu_include_dir}/torch_npu/csrc/core/npu/NPUGuard.h")
    if(NOT EXISTS "${_torch_npu_stream_header}" OR
       NOT EXISTS "${_torch_npu_guard_header}" OR
       NOT IS_DIRECTORY "${_torch_npu_library_dir}")
        message(FATAL_ERROR
            "Could not find the installed `torch_npu` C++ development files.")
    endif()

    file(STRINGS "${_torch_npu_stream_header}" _torch_npu_external_stream_api
        REGEX "getStreamFromExternal")
    if(NOT _torch_npu_external_stream_api)
        message(FATAL_ERROR
            "torch_npu ${_torch_npu_version} does not provide "
            "`c10_npu::getStreamFromExternal`; install a compatible release "
            "such as torch_npu 2.10.0.post4 or newer.")
    endif()

    find_library(TORCH_NPU_LIB torch_npu
        HINTS "${_torch_npu_library_dir}"
        NO_DEFAULT_PATH REQUIRED)
    list(APPEND TORCH_INCLUDE_DIRS "${_torch_npu_include_dir}")
    list(APPEND TORCH_LIBRARIES "${TORCH_NPU_LIB}")
    list(APPEND TORCH_RUNTIME_DIRS "${_torch_npu_library_dir}")
    message(STATUS "Found torch_npu ${_torch_npu_version}: ${TORCH_NPU_LIB}")
endif()

if(WITH_CAMBRICON AND TORCH_SOURCES)
    execute_process(
        COMMAND "${_TORCH_PYTHON}" -c
            "import importlib.metadata, pathlib; root = pathlib.Path(importlib.metadata.distribution('torch-mlu').locate_file('')).resolve(); include = root / 'torch_mlu/csrc'; lib = include / 'lib'; print(include, lib, sep=';')"
        OUTPUT_VARIABLE _torch_mlu_paths
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _torch_mlu_paths_result
    )
    if(NOT _torch_mlu_paths_result EQUAL 0 OR NOT _torch_mlu_paths)
        message(FATAL_ERROR
            "Could not determine `torch-mlu` include and library directories.")
    endif()
    list(GET _torch_mlu_paths 0 _torch_mlu_include_dir)
    list(GET _torch_mlu_paths 1 _torch_mlu_library_dir)
    if(NOT IS_DIRECTORY "${_torch_mlu_include_dir}" OR
       NOT IS_DIRECTORY "${_torch_mlu_library_dir}")
        message(FATAL_ERROR
            "Could not find the installed `torch-mlu` C++ development files.")
    endif()

    find_library(TORCH_MLU_LIB torch_mlu
        HINTS "${_torch_mlu_library_dir}"
        NO_DEFAULT_PATH REQUIRED)
    list(APPEND TORCH_INCLUDE_DIRS "${_torch_mlu_include_dir}")
    list(APPEND TORCH_LIBRARIES "${TORCH_MLU_LIB}")
    list(APPEND TORCH_RUNTIME_DIRS "${_torch_mlu_library_dir}")
endif()

if(TORCH_SOURCES)
    set(INFINI_OPS_TORCH_UNITY_BATCH_SIZE "8" CACHE STRING
        "Number of torch sources to include in each generated unity translation unit; set to 1 to disable")
    set(TORCH_COMPILE_SOURCES ${TORCH_SOURCES})
    if(INFINI_OPS_TORCH_UNITY_BATCH_SIZE GREATER 1)
        set(_torch_unity_dir "${CMAKE_CURRENT_BINARY_DIR}/torch_unity")
        file(MAKE_DIRECTORY "${_torch_unity_dir}")

        set(TORCH_COMPILE_SOURCES)
        set(_torch_unity_expected)
        set(_torch_unity_index 0)
        set(_torch_unity_count 0)
        set(_torch_unity_content "")
        foreach(_src IN LISTS TORCH_SOURCES)
            if(_torch_unity_count EQUAL 0)
                set(_torch_unity_src
                    "${_torch_unity_dir}/torch_unity_${_torch_unity_index}.cc")
                list(APPEND TORCH_COMPILE_SOURCES "${_torch_unity_src}")
                list(APPEND _torch_unity_expected "${_torch_unity_src}")
                set(_torch_unity_content
                    "// Generated by CMake to batch ATen-heavy torch wrappers.\n")
            endif()

            string(APPEND _torch_unity_content "#include \"${_src}\"\n")

            math(EXPR _torch_unity_count "${_torch_unity_count} + 1")
            if(_torch_unity_count GREATER_EQUAL INFINI_OPS_TORCH_UNITY_BATCH_SIZE)
                _infini_ops_write_if_different(
                    "${_torch_unity_src}" "${_torch_unity_content}")
                math(EXPR _torch_unity_index "${_torch_unity_index} + 1")
                set(_torch_unity_count 0)
                set(_torch_unity_content "")
            endif()
        endforeach()
        if(NOT _torch_unity_count EQUAL 0)
            _infini_ops_write_if_different(
                "${_torch_unity_src}" "${_torch_unity_content}")
        endif()
        _infini_ops_prune_stale_cc_files(
            "${_torch_unity_dir}" EXPECTED ${_torch_unity_expected})

        list(LENGTH TORCH_SOURCES _torch_source_count)
        list(LENGTH TORCH_COMPILE_SOURCES _torch_unity_source_count)
        message(STATUS
            "Torch unity build: ${_torch_source_count} sources batched into "
            "${_torch_unity_source_count} translation units")
    endif()

    if(WITH_TORCH)
        target_compile_definitions(infiniops PUBLIC
            $<BUILD_INTERFACE:WITH_TORCH=1>)
    endif()
    target_link_libraries(infiniops PUBLIC
        "$<BUILD_INTERFACE:${TORCH_LIBRARIES}>")
    target_include_directories(infiniops PUBLIC
        "$<BUILD_INTERFACE:${TORCH_INCLUDE_DIRS}>"
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/generated>
    )

    # Each generated `<op>.cc` instantiates `at::<op>_out(...)`, which
    # pulls in roughly 0.5-1 GB of ATen template metaprogramming.  At
    # ninja's default parallelism (one job per CPU), a build with 451
    # ops can blow past 30 GB of RSS and the OOM killer drops
    # `cc1plus`.  Cap the heavyweight torch sources via a Ninja job pool;
    # the rest of the build keeps full parallelism.  CI can raise this when
    # the container has enough memory.
    set(INFINI_OPS_TORCH_COMPILE_JOBS "2" CACHE STRING
        "Maximum concurrent generated torch source compilations")
    if(CMAKE_GENERATOR MATCHES "Ninja")
        set_property(GLOBAL APPEND PROPERTY JOB_POOLS
            torch_compile=${INFINI_OPS_TORCH_COMPILE_JOBS})
    endif()

    if(WITH_METAX OR WITH_MOORE OR WITH_MARS)
        # Vendor compilers (`mxcc`/`mcc`/`htcc`) cannot compile vendor-forked
        # `torch` headers. Compile `torch` sources with the system C++ compiler
        # instead.
        find_program(SYSTEM_CXX NAMES g++ c++)

        if(NOT SYSTEM_CXX)
            message(FATAL_ERROR "Could not find system `g++` for `torch` compilation.")
        endif()

        set(_torch_include_flags "")
        foreach(_dir ${TORCH_INCLUDE_DIRS})
            list(APPEND _torch_include_flags "-isystem" "${_dir}")
        endforeach()

        set(_torch_vendor_include_flags "")
        if(WITH_METAX)
            list(APPEND _torch_vendor_include_flags
                "-I${MACA_PATH}/include"
                "-I${MACA_PATH}/include/mcr"
                "-I${MACA_PATH}/tools/cu-bridge/include")
        endif()
        if(WITH_MARS)
            list(APPEND _torch_vendor_include_flags
                "-I${HPCC_PATH}/include"
                "-I${HPCC_PATH}/tools/cu-bridge/include")
        endif()
        if(WITH_MOORE)
            list(APPEND _torch_vendor_include_flags "-I${MUSA_ROOT}/include")
            execute_process(
                COMMAND "${_TORCH_PYTHON}" -c
                    "import importlib.metadata, pathlib; root = pathlib.Path(importlib.metadata.distribution('torch-musa').locate_file('')).resolve(); compat = root / 'torch_musa/share/generated_cuda_compatible/include'; lib = root / 'torch_musa/lib'; print(root, compat, lib, sep=';')"
                OUTPUT_VARIABLE _torch_musa_paths
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE _torch_musa_paths_result
            )
            if(NOT _torch_musa_paths_result EQUAL 0 OR NOT _torch_musa_paths)
                message(FATAL_ERROR
                    "Could not determine `torch_musa` include and library directories.")
            endif()
            list(GET _torch_musa_paths 2 _torch_musa_library_dir)
            list(REMOVE_AT _torch_musa_paths 2)
            foreach(_dir IN LISTS _torch_musa_paths)
                if(NOT IS_DIRECTORY "${_dir}")
                    message(FATAL_ERROR
                        "Could not find `torch_musa` include directory `${_dir}`.")
                endif()
                list(APPEND _torch_include_flags "-isystem" "${_dir}")
            endforeach()

            find_library(TORCH_MUSA_LIB musa_python
                HINTS "${_torch_musa_library_dir}"
                NO_DEFAULT_PATH REQUIRED)
            target_link_libraries(infiniops PUBLIC
                $<BUILD_INTERFACE:${TORCH_MUSA_LIB}>)
            list(APPEND TORCH_RUNTIME_DIRS "${_torch_musa_library_dir}")
        endif()

        # Vendor-specific defines required by forked `torch` headers.
        set(_torch_extra_flags "")
        if(WITH_METAX)
            list(APPEND _torch_extra_flags "-DUSE_MACA=1" "-DWITH_METAX=1")
        endif()
        if(WITH_MARS)
            list(APPEND _torch_extra_flags "-DUSE_HPCC=1" "-DWITH_MARS=1")
        endif()
        if(WITH_MOORE)
            list(APPEND _torch_extra_flags "-DWITH_MOORE=1")
        endif()
        if(WITH_CPU)
            list(APPEND _torch_extra_flags "-DWITH_CPU=1")
        endif()
        if(INFINI_OPS_ENABLE_HOST_RANGE_PROFILING)
            list(APPEND _torch_extra_flags
                "-DINFINI_OPS_ENABLE_HOST_RANGE_PROFILING=1")
        endif()
        if(DEFINED TORCH_CXX11_ABI)
            list(APPEND _torch_extra_flags "-D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}")
        endif()

        set(TORCH_OBJECT_DIR "${CMAKE_CURRENT_BINARY_DIR}/torch_objs")
        file(MAKE_DIRECTORY "${TORCH_OBJECT_DIR}")

        set(TORCH_OBJECT_FILES)
        foreach(_src ${TORCH_COMPILE_SOURCES})
            file(RELATIVE_PATH _rel ${CMAKE_CURRENT_SOURCE_DIR} ${_src})
            string(REPLACE "/" "_" _obj_name "${_rel}")
            string(REPLACE ".cc" ".o" _obj_name "${_obj_name}")
            string(REPLACE ".cpp" ".o" _obj_name "${_obj_name}")
            set(_obj "${TORCH_OBJECT_DIR}/${_obj_name}")
            set(_depfile "${_obj}.d")

            add_custom_command(
                OUTPUT "${_obj}"
                COMMAND ${SYSTEM_CXX}
                    -std=c++17 -fPIC -O0
                    "-I${CMAKE_CURRENT_SOURCE_DIR}"
                    "-I${PROJECT_SOURCE_DIR}/generated"
                    ${INFINI_RT_INCLUDE_FLAGS}
                    ${_torch_vendor_include_flags}
                    ${_torch_include_flags}
                    ${_torch_extra_flags}
                    -MMD -MT "${_obj}" -MF "${_depfile}"
                    -c "${_src}" -o "${_obj}"
                DEPENDS "${_src}"
                DEPFILE "${_depfile}"
                COMMENT "Compiling ${_rel} with system C++ compiler"
                JOB_POOL torch_compile
            )
            list(APPEND TORCH_OBJECT_FILES "${_obj}")
        endforeach()

        set_source_files_properties(${TORCH_OBJECT_FILES}
            PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
        target_sources(infiniops PRIVATE ${TORCH_OBJECT_FILES})
    else()
        # Build the heavy torch sources as their own object library so
        # the Ninja `torch_compile` job pool throttles only those
        # compilations and the rest of `infiniops` keeps full
        # parallelism.  Inherit infiniops's compile-time settings via
        # generator expressions (linking would create a cyclic
        # dependency since infiniops then absorbs the object files).
        add_library(infini_ops_torch_objs OBJECT ${TORCH_COMPILE_SOURCES})
        set_target_properties(infini_ops_torch_objs
            PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_include_directories(infini_ops_torch_objs PRIVATE
            $<TARGET_PROPERTY:infiniops,INCLUDE_DIRECTORIES>
            ${INFINI_RT_INCLUDE_DIRS}
            ${TORCH_INCLUDE_DIRS}
            ${PROJECT_SOURCE_DIR}/generated)
        target_compile_definitions(infini_ops_torch_objs PRIVATE
            $<TARGET_PROPERTY:infiniops,COMPILE_DEFINITIONS>)
        target_compile_options(infini_ops_torch_objs PRIVATE
            $<TARGET_PROPERTY:infiniops,COMPILE_OPTIONS>
            -O0)
        if(WITH_HYGON)
            # DTK's CUDA compatibility headers and PyTorch HIP headers share
            # vector types. Include HIP first and select their AMD bridge.
            target_compile_definitions(infini_ops_torch_objs PRIVATE
                __HIP_PLATFORM_AMD__ __CUDA_HIP_PLATFORM_AMD__)
            target_compile_options(infini_ops_torch_objs PRIVATE
                -include hip/hip_runtime.h)
        endif()
        if(CMAKE_GENERATOR MATCHES "Ninja")
            set_target_properties(infini_ops_torch_objs
                PROPERTIES JOB_POOL_COMPILE torch_compile)
        endif()
        target_sources(infiniops PRIVATE
            $<TARGET_OBJECTS:infini_ops_torch_objs>)
    endif()
endif()

set(_INFINI_OPS_LINKED_INSTALL_RPATH "$ORIGIN")
if(WITH_LINKED)
    if(_infini_ops_linked_uses_torch)
        list(APPEND _INFINI_OPS_LINKED_INSTALL_RPATH ${TORCH_RUNTIME_DIRS})
    endif()
    list(APPEND _INFINI_OPS_LINKED_INSTALL_RPATH
        ${INFINI_OPS_LINKED_RUNTIME_DIRS})
    list(REMOVE_DUPLICATES _INFINI_OPS_LINKED_INSTALL_RPATH)
endif()

if(WITH_LINKED OR NOT GENERATE_PYTHON_BINDINGS)
    set_target_properties(infiniops PROPERTIES
        INSTALL_RPATH "${_INFINI_OPS_LINKED_INSTALL_RPATH}")
endif()

target_include_directories(infiniops
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/generated/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/generated/bindings>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/operator.h"
    "${PROJECT_SOURCE_DIR}/scripts/generate_public_operator_header.py"
    "${PROJECT_SOURCE_DIR}/scripts/generate_wrappers.py"
)

if(GENERATE_OPERATOR_CALL_INSTANTIATIONS OR GENERATE_PYTHON_BINDINGS)
    find_package(Python COMPONENTS Interpreter REQUIRED)
    # Always regenerate wrappers so emitted call instantiations and bindings
    # match the active device list.  Stale generated files would omit
    # specializations for enabled backends, causing link-time or runtime
    # failures.

    set(GENERATOR_ARGS --devices ${DEVICE_LIST})
    if(_infini_ops_ops_config)
        list(APPEND GENERATOR_ARGS
            --ops-config "${_infini_ops_ops_config}")
    endif()
    if(_infini_ops_op_allowlist)
        list(APPEND GENERATOR_ARGS --ops ${_infini_ops_op_allowlist})
    endif()
    if(WITH_TORCH)
        list(APPEND GENERATOR_ARGS --with-torch)
    endif()
    if(WITH_LINKED)
        list(APPEND GENERATOR_ARGS --with-linked)
    endif()
    if(BUILD_NVIDIA_FLASH_INFERENCE)
        list(APPEND GENERATOR_ARGS --with-native-flash)
    endif()
    if(BUILD_NVIDIA_W4_MARLIN)
        list(APPEND GENERATOR_ARGS --with-native-w4)
    endif()
    if(WITH_NINETOOTHED)
        list(APPEND GENERATOR_ARGS --with-ninetoothed)
    endif()

    if(WITH_TRITON)
        list(APPEND GENERATOR_ARGS --with-triton)
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env
                INFINI_RT_INCLUDE_DIRS=${INFINI_RT_INCLUDE_DIRS_ENV}
                ${Python_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/scripts/generate_wrappers.py
                ${GENERATOR_ARGS}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        RESULT_VARIABLE script_result
    )

    if(NOT script_result EQUAL 0)
        message(FATAL_ERROR "Generating wrappers - failed")
    else()
        message(STATUS "Generating wrappers - done")
    endif()
endif()

find_package(Python COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND ${Python_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/scripts/generate_public_operator_header.py
        --source ${CMAKE_CURRENT_SOURCE_DIR}/operator.h
        --output ${PROJECT_SOURCE_DIR}/generated/include/operator.h
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    RESULT_VARIABLE public_operator_header_result
)
if(NOT public_operator_header_result EQUAL 0)
    message(FATAL_ERROR "Generating public operator header - failed")
endif()
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/config.h
    ${PROJECT_SOURCE_DIR}/generated/include/config.h
    COPYONLY
)

if(GENERATE_OPERATOR_CALL_INSTANTIATIONS)
    target_sources(infiniops PRIVATE
        "${PROJECT_SOURCE_DIR}/generated/src/cache_clear_dispatch.cc")

    file(GLOB_RECURSE OPERATOR_CALL_INSTANTIATION_SOURCES CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/generated/src/operator_call_instantiations_*.cc")

    if(WITH_NVIDIA OR WITH_HYGON OR WITH_THEAD)
        set_source_files_properties(${OPERATOR_CALL_INSTANTIATION_SOURCES}
            PROPERTIES LANGUAGE CUDA)
        target_sources(infiniops PRIVATE ${OPERATOR_CALL_INSTANTIATION_SOURCES})
    elseif(WITH_ILUVATAR)
        set(_iluvatar_call_instantiation_include_flags
            "-I${CMAKE_CURRENT_SOURCE_DIR}"
            "-I${PROJECT_SOURCE_DIR}"
            "-I${PROJECT_SOURCE_DIR}/generated"
            "-I${PROJECT_SOURCE_DIR}/generated/include"
            ${INFINI_RT_INCLUDE_FLAGS})
        foreach(_dir IN LISTS TORCH_INCLUDE_DIRS CUDAToolkit_INCLUDE_DIRS)
            list(APPEND _iluvatar_call_instantiation_include_flags "-I${_dir}")
        endforeach()

        set(_iluvatar_call_instantiation_defs
            -DWITH_ILUVATAR=1
            -DINFINI_OPS_USE_OPERATOR_CALL_INSTANTIATIONS=1)
        if(INFINI_OPS_ENABLE_HOST_RANGE_PROFILING)
            list(APPEND _iluvatar_call_instantiation_defs
                -DINFINI_OPS_ENABLE_HOST_RANGE_PROFILING=1)
        endif()
        if(WITH_CPU)
            list(APPEND _iluvatar_call_instantiation_defs -DWITH_CPU=1)
        endif()
        if(WITH_TORCH)
            list(APPEND _iluvatar_call_instantiation_defs -DWITH_TORCH=1)
        endif()
        if(DEFINED TORCH_CXX11_ABI)
            list(APPEND _iluvatar_call_instantiation_defs
                "-D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}")
        endif()

        set(ILUVATAR_CALL_INSTANTIATION_OBJECTS)
        set(_iluvatar_call_instantiation_object_dir
            "${CMAKE_CURRENT_BINARY_DIR}/iluvatar_call_instantiation_objs")
        foreach(_src IN LISTS OPERATOR_CALL_INSTANTIATION_SOURCES)
            get_filename_component(_name "${_src}" NAME_WE)
            set(_obj "${_iluvatar_call_instantiation_object_dir}/${_name}.o")
            set(_dep "${_obj}.d")
            set(_depfile_arg)
            if(CMAKE_GENERATOR MATCHES "Ninja")
                set(_depfile_arg DEPFILE "${_dep}")
            endif()
            add_custom_command(
                OUTPUT "${_obj}"
                COMMAND ${CMAKE_COMMAND} -E make_directory
                    "${_iluvatar_call_instantiation_object_dir}"
                COMMAND ${ILUVATAR_CUDA_COMPILER}
                    ${_iluvatar_call_instantiation_defs}
                    ${_iluvatar_call_instantiation_include_flags}
                    ${ILUVATAR_CUDA_FLAGS}
                    -MMD -MF "${_dep}"
                    -c "${_src}" -o "${_obj}"
                DEPENDS "${_src}"
                ${_depfile_arg}
                COMMENT "Compiling ${_name}.cc with CoreX clang++"
                VERBATIM
            )
            list(APPEND ILUVATAR_CALL_INSTANTIATION_OBJECTS "${_obj}")
        endforeach()

        set_source_files_properties(${ILUVATAR_CALL_INSTANTIATION_OBJECTS}
            PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
        target_sources(infiniops PRIVATE ${ILUVATAR_CALL_INSTANTIATION_OBJECTS})
    else()
        target_sources(infiniops PRIVATE ${OPERATOR_CALL_INSTANTIATION_SOURCES})
    endif()
endif()

if(GENERATE_OPERATOR_CALL_INSTANTIATIONS)
    file(GLOB_RECURSE OPERATOR_DISPATCH_SOURCES CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/generated/bindings/generated_dispatch_*.cc")

    if(OPERATOR_DISPATCH_SOURCES)
        target_include_directories(infiniops PRIVATE "${PROJECT_SOURCE_DIR}")
        if(WITH_NVIDIA OR WITH_HYGON OR WITH_THEAD)
            set_source_files_properties(${OPERATOR_DISPATCH_SOURCES}
                PROPERTIES LANGUAGE CUDA)
            target_sources(infiniops PRIVATE ${OPERATOR_DISPATCH_SOURCES})
        elseif(WITH_ILUVATAR)
            set(ILUVATAR_OPERATOR_DISPATCH_OBJECTS)
            set(_iluvatar_operator_dispatch_object_dir
                "${CMAKE_CURRENT_BINARY_DIR}/iluvatar_operator_dispatch_objs")
            set(_iluvatar_operator_dispatch_include_flags
                ${_iluvatar_call_instantiation_include_flags}
                "-I${PROJECT_SOURCE_DIR}/generated/bindings")
            foreach(_src IN LISTS OPERATOR_DISPATCH_SOURCES)
                get_filename_component(_name "${_src}" NAME_WE)
                set(_obj "${_iluvatar_operator_dispatch_object_dir}/${_name}.o")
                set(_dep "${_obj}.d")
                set(_depfile_arg)
                if(CMAKE_GENERATOR MATCHES "Ninja")
                    set(_depfile_arg DEPFILE "${_dep}")
                endif()
                add_custom_command(
                    OUTPUT "${_obj}"
                    COMMAND ${CMAKE_COMMAND} -E make_directory
                        "${_iluvatar_operator_dispatch_object_dir}"
                    COMMAND ${ILUVATAR_CUDA_COMPILER}
                        ${_iluvatar_call_instantiation_defs}
                        ${_iluvatar_operator_dispatch_include_flags}
                        ${ILUVATAR_CUDA_FLAGS}
                        -MMD -MF "${_dep}"
                        -c "${_src}" -o "${_obj}"
                    DEPENDS "${_src}"
                    ${_depfile_arg}
                    COMMENT "Compiling ${_name}.cc with CoreX clang++"
                    VERBATIM
                )
                list(APPEND ILUVATAR_OPERATOR_DISPATCH_OBJECTS "${_obj}")
            endforeach()

            set_source_files_properties(${ILUVATAR_OPERATOR_DISPATCH_OBJECTS}
                PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
            target_sources(infiniops PRIVATE ${ILUVATAR_OPERATOR_DISPATCH_OBJECTS})
        endif()
    endif()
endif()

if(GENERATE_PYTHON_BINDINGS)
    file(GLOB_RECURSE PYBIND11_SOURCES CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/generated/bindings/*.cc")

    set(PYBIND11_DISPATCH_SOURCES)
    set(PYBIND11_UNITY_SOURCES)
    foreach(_src IN LISTS PYBIND11_SOURCES)
        if(_src MATCHES "/generated_dispatch[^/]*\\.cc$")
            list(APPEND PYBIND11_DISPATCH_SOURCES "${_src}")
        else()
            list(APPEND PYBIND11_UNITY_SOURCES "${_src}")
        endif()
    endforeach()

    set(PYBIND11_COMPILE_SOURCES ${PYBIND11_UNITY_SOURCES})
    if(WITH_TORCH)
        set(INFINI_OPS_BINDING_UNITY_BATCH_SIZE "8" CACHE STRING
            "Number of generated pybind11 sources to include in each unity translation unit; set to 1 to disable")
        if(INFINI_OPS_BINDING_UNITY_BATCH_SIZE GREATER 1)
            set(_binding_unity_dir "${CMAKE_CURRENT_BINARY_DIR}/binding_unity")
            file(MAKE_DIRECTORY "${_binding_unity_dir}")

            set(PYBIND11_COMPILE_SOURCES)
            set(_binding_unity_expected)
            set(_binding_unity_index 0)
            set(_binding_unity_count 0)
            set(_binding_unity_content "")
            foreach(_src IN LISTS PYBIND11_UNITY_SOURCES)
                if(_binding_unity_count EQUAL 0)
                    set(_binding_unity_src
                        "${_binding_unity_dir}/binding_unity_${_binding_unity_index}.cc")
                    list(APPEND PYBIND11_COMPILE_SOURCES "${_binding_unity_src}")
                    list(APPEND _binding_unity_expected "${_binding_unity_src}")
                    set(_binding_unity_content
                        "// Generated by CMake to batch pybind11 wrapper sources.\n")
                endif()

                string(APPEND _binding_unity_content "#include \"${_src}\"\n")

                math(EXPR _binding_unity_count "${_binding_unity_count} + 1")
                if(_binding_unity_count GREATER_EQUAL INFINI_OPS_BINDING_UNITY_BATCH_SIZE)
                    _infini_ops_write_if_different(
                        "${_binding_unity_src}" "${_binding_unity_content}")
                    math(EXPR _binding_unity_index "${_binding_unity_index} + 1")
                    set(_binding_unity_count 0)
                    set(_binding_unity_content "")
                endif()
            endforeach()
            if(NOT _binding_unity_count EQUAL 0)
                _infini_ops_write_if_different(
                    "${_binding_unity_src}" "${_binding_unity_content}")
            endif()
            _infini_ops_prune_stale_cc_files(
                "${_binding_unity_dir}" EXPECTED ${_binding_unity_expected})

            list(LENGTH PYBIND11_UNITY_SOURCES _binding_source_count)
            list(LENGTH PYBIND11_COMPILE_SOURCES _binding_unity_source_count)
            message(STATUS
                "Binding unity build: ${_binding_source_count} sources batched into "
                "${_binding_unity_source_count} translation units")
        endif()
    endif()
    list(APPEND PYBIND11_COMPILE_SOURCES ${PYBIND11_DISPATCH_SOURCES})

    if(WITH_NVIDIA OR WITH_HYGON OR WITH_THEAD)
        set_source_files_properties(${PYBIND11_COMPILE_SOURCES} PROPERTIES LANGUAGE CUDA)
    elseif(WITH_ILUVATAR)
        set(_iluvatar_dispatch_include_flags
            "-I${CMAKE_CURRENT_SOURCE_DIR}"
            "-I${PROJECT_SOURCE_DIR}"
            "-I${PROJECT_SOURCE_DIR}/generated"
            ${INFINI_RT_INCLUDE_FLAGS})
        foreach(_dir IN LISTS TORCH_INCLUDE_DIRS CUDAToolkit_INCLUDE_DIRS)
            list(APPEND _iluvatar_dispatch_include_flags "-I${_dir}")
        endforeach()

        set(_iluvatar_dispatch_defs -DWITH_ILUVATAR=1)
        if(INFINI_OPS_ENABLE_HOST_RANGE_PROFILING)
            list(APPEND _iluvatar_dispatch_defs
                -DINFINI_OPS_ENABLE_HOST_RANGE_PROFILING=1)
        endif()
        if(GENERATE_OPERATOR_CALL_INSTANTIATIONS)
            list(APPEND _iluvatar_dispatch_defs
                -DINFINI_OPS_USE_OPERATOR_CALL_INSTANTIATIONS=1)
        endif()
        if(WITH_TORCH)
            list(APPEND _iluvatar_dispatch_defs -DWITH_TORCH=1)
        endif()
        if(DEFINED TORCH_CXX11_ABI)
            list(APPEND _iluvatar_dispatch_defs "-D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}")
        endif()

        set(ILUVATAR_DISPATCH_OBJECTS)
        set(_iluvatar_dispatch_object_dir "${CMAKE_CURRENT_BINARY_DIR}/iluvatar_dispatch_objs")
        foreach(_src IN LISTS PYBIND11_DISPATCH_SOURCES)
            get_filename_component(_name "${_src}" NAME_WE)
            set(_obj "${_iluvatar_dispatch_object_dir}/${_name}.o")
            set(_dep "${_obj}.d")
            set(_depfile_arg)
            if(CMAKE_GENERATOR MATCHES "Ninja")
                set(_depfile_arg DEPFILE "${_dep}")
            endif()
            add_custom_command(
                OUTPUT "${_obj}"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${_iluvatar_dispatch_object_dir}"
                COMMAND ${ILUVATAR_CUDA_COMPILER}
                    ${_iluvatar_dispatch_defs}
                    ${_iluvatar_dispatch_include_flags}
                    ${ILUVATAR_CUDA_FLAGS}
                    -MMD -MF "${_dep}"
                    -c "${_src}" -o "${_obj}"
                DEPENDS "${_src}"
                ${_depfile_arg}
                COMMENT "Compiling ${_name}.cc with CoreX clang++"
                VERBATIM
            )
            list(APPEND ILUVATAR_DISPATCH_OBJECTS "${_obj}")
        endforeach()

        list(REMOVE_ITEM PYBIND11_COMPILE_SOURCES ${PYBIND11_DISPATCH_SOURCES})
        set_source_files_properties(${ILUVATAR_DISPATCH_OBJECTS}
            PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
        list(APPEND PYBIND11_COMPILE_SOURCES ${ILUVATAR_DISPATCH_OBJECTS})
    endif()

    if(PYBIND11_ENABLE_EXTRAS)
        pybind11_add_module(ops ${PYBIND11_COMPILE_SOURCES})
    else()
        pybind11_add_module(ops NO_EXTRAS ${PYBIND11_COMPILE_SOURCES})
    endif()

    target_compile_options(ops PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX,CUDA>:-Wno-deprecated-declarations>"
    )

    if(WITH_CAMBRICON AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(ops PRIVATE
            "${_cambricon_host_bang_compat_option}")
    endif()

    if(GENERATE_OPERATOR_CALL_INSTANTIATIONS)
        target_compile_definitions(ops PRIVATE
            INFINI_OPS_USE_OPERATOR_CALL_INSTANTIATIONS=1)
    endif()

    if(WITH_TORCH AND CMAKE_GENERATOR MATCHES "Ninja")
        set(INFINI_OPS_BINDING_COMPILE_JOBS "2" CACHE STRING
            "Maximum concurrent generated pybind11 binding compilations")
        set_property(GLOBAL APPEND PROPERTY JOB_POOLS
            binding_compile=${INFINI_OPS_BINDING_COMPILE_JOBS})
        set_property(TARGET ops PROPERTY JOB_POOL_COMPILE binding_compile)
    endif()

    if(WITH_METAX)
        target_compile_options(ops PRIVATE "-x" "maca")
    endif()
    if(WITH_MOORE)
        target_compile_options(ops PRIVATE "-x" "musa")
    endif()

    if(WITH_HYGON)
        set_target_properties(ops PROPERTIES
            RULE_LAUNCH_COMPILE "${_HYGON_RULE_LAUNCH_ENV}"
            RULE_LAUNCH_LINK "${_HYGON_RULE_LAUNCH_ENV}"
        )
    endif()

    target_include_directories(ops PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/generated/include
    )
    if(WITH_NINETOOTHED)
        target_include_directories(ops PRIVATE
            ${INFINI_OPS_NINETOOTHED_INCLUDE_DIRS})
    endif()

    target_link_libraries(ops PRIVATE infiniops)

    # Cambricon generated dispatch is compiled into the Python extension and
    # directly references host launch stubs emitted from `.mlu` sources. Link
    # those objects into `ops` too so import does not depend on re-exporting
    # internal C++ symbols from `libinfiniops`.
    if(WITH_CAMBRICON AND CAMBRICON_OBJECT_FILES)
        target_sources(ops PRIVATE ${CAMBRICON_OBJECT_FILES})
    endif()

    # Custom `AscendC` kernel objects must be linked directly into ops
    # because the `AscendC` toolchain compiles host stubs with hidden
    # visibility — `libinfiniops.so` cannot re-export those symbols.
    # The `Operator<..., 1>` template instantiations that call
    # `aclrtlaunch_*` live in `ops.cc`, so link here with
    # `--whole-archive` to ensure all launch functions are available.
    # `$<TARGET_FILE>` works for both real `ascendc_library()` targets and
    # `IMPORTED` targets pointing at a pre-built `.a`.  The
    # `no_workspace_kernel` target is only created inside the
    # `WITH_ASCEND` block above, so this branch must mirror that gate;
    # otherwise non-Ascend builds error out with "No target
    # no_workspace_kernel".
    if(WITH_ASCEND AND BUILD_ASCEND_CUSTOM)
        target_include_directories(ops PRIVATE
            "${_custom_build_dir}/include/no_workspace_kernel")
        target_link_libraries(ops PRIVATE
            -Wl,--whole-archive $<TARGET_FILE:no_workspace_kernel> -Wl,--no-whole-archive)
        # `ops` link step must wait for `build.sh` to produce the `.a`.
        add_dependencies(ops no_workspace_kernel_build)
    endif()
    set(_INFINI_OPS_INSTALL_RPATH "$ORIGIN")
    if(WITH_LINKED)
        set(_INFINI_OPS_INSTALL_RPATH
            ${_INFINI_OPS_LINKED_INSTALL_RPATH})
    elseif(WITH_TORCH)
        list(APPEND _INFINI_OPS_INSTALL_RPATH ${TORCH_RUNTIME_DIRS})
    endif()

    set_target_properties(infiniops PROPERTIES
        INSTALL_RPATH "${_INFINI_OPS_INSTALL_RPATH}")
    set_target_properties(ops PROPERTIES
        INSTALL_RPATH "${_INFINI_OPS_INSTALL_RPATH}")

    get_target_property(_infinirt_imported infinirt IMPORTED)
    if(_infinirt_imported)
        install(TARGETS infiniops ops DESTINATION .)
        get_target_property(_infinirt_imported_location infinirt IMPORTED_LOCATION)
        install(FILES "${_infinirt_imported_location}" DESTINATION .)
    else()
        install(TARGETS infinirt infiniops ops DESTINATION .)
    endif()

    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/__init__.py" "")
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/__init__.py" DESTINATION .)

    if(WITH_TORCH)
        # Ship the per-op metadata alongside the bindings so the unified
        # torch op test can discover what to exercise at runtime.
        install(FILES "${PROJECT_SOURCE_DIR}/generated/torch_ops_metadata.json"
            DESTINATION .)
    endif()

    if(WITH_TRITON)
        # Ship the JIT compiler and kernel sources so Triton JIT operators
        # can compile kernels at runtime. `compile.py` uses `__file__` to
        # locate `ops/` relative to itself; both must live under `triton/`.
        install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/triton/jit/compile.py"
                DESTINATION triton/jit)
        install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/triton/ops/"
                DESTINATION triton/ops
                FILES_MATCHING
                PATTERN "*.py")
    endif()
endif()

install(TARGETS infiniops
    EXPORT InfiniOpsTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES "${PROJECT_SOURCE_DIR}/LICENSE" "${PROJECT_SOURCE_DIR}/THIRD_PARTY.txt"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/InfiniOps")
install(FILES "${PROJECT_SOURCE_DIR}/LICENSES/InfiniCore-MIT.txt"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/InfiniOps")
if(BUILD_NVIDIA_W4_MARLIN)
    install(FILES
        "${PROJECT_SOURCE_DIR}/LICENSES/FasterTransformer-BSD-3-Clause.txt"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/InfiniOps")
endif()
if(WITH_NVIDIA)
    install(FILES "${PROJECT_SOURCE_DIR}/LICENSES/CUTLASS-BSD-3-Clause.txt"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/InfiniOps")
endif()
if(WITH_LINKED)
    if(_infini_ops_linked_uses_tvm_ffi)
        # Fixed native sampling/runtime recipe; not the complete Python wheels.
        install(FILES "${PROJECT_SOURCE_DIR}/LICENSES/Apache-2.0-components.txt"
            DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/InfiniOps")
        install(FILES "${PROJECT_SOURCE_DIR}/LICENSES/FlashInfer-0.6.7/NOTICE.txt"
            DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/InfiniOps/FlashInfer-0.6.7")
        install(FILES "${PROJECT_SOURCE_DIR}/LICENSES/TVM-FFI-0.1.10/NOTICE.txt"
            "${PROJECT_SOURCE_DIR}/LICENSES/TVM-FFI-0.1.10/LICENSE.libbacktrace.txt"
            DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/InfiniOps/TVM-FFI-0.1.10")
    endif()
    # Keep the real inventory for resolved packages, including unknown versions.
    foreach(_notice IN LISTS INFINI_OPS_LINKED_NOTICE_FILES)
        get_filename_component(_notice_dir "${_notice}" DIRECTORY)
        string(SHA256 _notice_group "${_notice_dir}")
        install(FILES "${_notice}"
            DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/InfiniOps/resolved/${_notice_group}")
    endforeach()
endif()

set(INFINI_OPS_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/InfiniOps")

install(EXPORT InfiniOpsTargets
    FILE InfiniOpsTargets.cmake
    NAMESPACE InfiniOps::
    DESTINATION ${INFINI_OPS_INSTALL_CMAKEDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/InfiniOpsConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/InfiniOpsConfig.cmake
    INSTALL_DESTINATION ${INFINI_OPS_INSTALL_CMAKEDIR}
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/InfiniOpsConfig.cmake
    DESTINATION ${INFINI_OPS_INSTALL_CMAKEDIR}
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(GENERATE_OPERATOR_CALL_INSTANTIATIONS)
    install(FILES
        ${PROJECT_SOURCE_DIR}/generated/include/infini/operator_call_instantiations.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/infini
    )
endif()

file(GLOB INFINI_OPS_PUBLIC_CORE_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
list(REMOVE_ITEM INFINI_OPS_PUBLIC_CORE_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/operator.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/host_range_profiler.h")

install(FILES
    ${PROJECT_SOURCE_DIR}/generated/include/operator.h
    ${INFINI_OPS_PUBLIC_CORE_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/base/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/base
    FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/common/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/common
    FILES_MATCHING PATTERN "*.h"
)

if(EXISTS ${PROJECT_SOURCE_DIR}/generated/base)
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/generated/base/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/base
        FILES_MATCHING PATTERN "*.h"
    )
endif()
