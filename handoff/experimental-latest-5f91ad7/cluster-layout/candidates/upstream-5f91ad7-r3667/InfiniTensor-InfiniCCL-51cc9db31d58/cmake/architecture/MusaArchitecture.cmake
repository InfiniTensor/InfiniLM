include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/MacroArchitecture.cmake")

# Normalize MUSA architecture spellings accepted from CMake cache, environment,
# or device probing; for example `mp_31`, `3.1`, and `31` all become `31`.
function(infiniccl_normalize_musa_architectures input output)
    infiniccl_normalize_architectures(
        "${input}"
        _architectures
        "mp_"
        "MUSA architecture"
    )
    set(${output} "${_architectures}" PARENT_SCOPE)
endfunction()

# Convert MUSA architectures to the two settings needed by Moore MCCL: `mcc`
# device flags such as `--offload-arch=mp_31`, and the `MARCH_TYPE` macro used
# by `mccl.h` to expose architecture-dependent datatypes.
function(infiniccl_compute_musa_architecture_config input architectures_output march_type_output flags_output)
    infiniccl_normalize_musa_architectures("${input}" _architectures)

    # `mccl.h` exposes data types globally from one `MARCH_TYPE`. Use the least
    # capable target so a fat binary never advertises a type that one of its
    # target architectures cannot execute.
    infiniccl_compute_least_capable_architecture_macro("${_architectures}" 10 _march_type)
    infiniccl_make_architecture_compile_options("${_architectures}" "--offload-arch=mp_" _flags)

    set(${architectures_output} "${_architectures}" PARENT_SCOPE)
    set(${march_type_output} "${_march_type}" PARENT_SCOPE)
    set(${flags_output} "${_flags}" PARENT_SCOPE)
endfunction()

# Resolve MUSA architecture settings in priority order: explicit
# `MUSA_ARCHITECTURES`, then `TORCH_MUSA_ARCH_LIST`, then installed GPU probing.
# The resolved values are exported for later target setup in `src` and examples.
function(infiniccl_resolve_musa_architecture_config musa_include_dir musart_library)
    set(_musa_architectures_help
        "MUSA GPU architectures (for example `31;22`); detected from installed GPUs when empty")
    set(MUSA_ARCHITECTURES "" CACHE STRING "${_musa_architectures_help}")

    set(_musa_architectures "${MUSA_ARCHITECTURES}")
    if(NOT _musa_architectures
       AND DEFINED ENV{TORCH_MUSA_ARCH_LIST}
       AND NOT "$ENV{TORCH_MUSA_ARCH_LIST}" STREQUAL "")
        set(_musa_architectures "$ENV{TORCH_MUSA_ARCH_LIST}")
        message(STATUS "Using MUSA architectures from `TORCH_MUSA_ARCH_LIST`.")
    endif()

    if(NOT _musa_architectures)
        infiniccl_detect_musa_architectures(
            "${musa_include_dir}"
            "${musart_library}"
            _musa_architectures
        )
        message(STATUS "Auto-detected MUSA architectures from installed GPUs.")
    endif()

    infiniccl_compute_musa_architecture_config(
        "${_musa_architectures}"
        _normalized_musa_architectures
        _musa_march_type
        _musa_arch_compile_options
    )

    set(MUSA_ARCHITECTURES "${_normalized_musa_architectures}"
        CACHE STRING "${_musa_architectures_help}" FORCE)
    set(MUSA_ARCHITECTURES "${_normalized_musa_architectures}" PARENT_SCOPE)
    set(MUSA_MARCH_TYPE "${_musa_march_type}" PARENT_SCOPE)
    set(MUSA_ARCH_COMPILE_OPTIONS "${_musa_arch_compile_options}" PARENT_SCOPE)
    message(STATUS
        "MUSA architectures: ${_normalized_musa_architectures} (`MARCH_TYPE=${_musa_march_type}`)")
endfunction()

# Add MUSA architecture settings to a target after it has been created, so only
# Moore/MUSA sources see `MARCH_TYPE` and `--offload-arch`.
function(infiniccl_configure_musa_target target)
    if(NOT TARGET ${target})
        message(FATAL_ERROR "Cannot configure unknown MUSA target `${target}`.")
    endif()

    infiniccl_configure_macro_architecture_target(
        ${target}
        MARCH_TYPE
        "${MUSA_MARCH_TYPE}"
        "${MUSA_ARCH_COMPILE_OPTIONS}"
    )
endfunction()

# Build and run a tiny MUSA runtime probe to return installed GPU capabilities
# as values like `31`, matching `musaDeviceProp.major/minor`.
function(infiniccl_detect_musa_architectures musa_include_dir musart_library output)
    set(_source "${CMAKE_CURRENT_BINARY_DIR}/get_musa_compute_capabilities.cpp")
    file(WRITE "${_source}" [=[
#include <musa_runtime.h>

#include <cstdio>

int main() {
  int device_count = 0;
  if (musaGetDeviceCount(&device_count) != musaSuccess || device_count == 0) {
    return 1;
  }

  for (int device = 0; device < device_count; ++device) {
    musaDeviceProp properties;
    if (musaGetDeviceProperties(&properties, device) != musaSuccess) {
      return 1;
    }
    std::printf("%d%d ", properties.major, properties.minor);
  }
  return 0;
}
]=])

    try_run(
        _run_result
        _compile_result
        "${CMAKE_CURRENT_BINARY_DIR}"
        "${_source}"
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${musa_include_dir}"
        LINK_LIBRARIES "${musart_library}"
        RUN_OUTPUT_VARIABLE _detected_architectures
    )

    if(NOT _compile_result OR NOT "${_run_result}" STREQUAL "0")
        message(FATAL_ERROR
            "Could not detect the architecture of the installed MUSA GPUs. "
            "Set `MUSA_ARCHITECTURES` explicitly, for example "
            "`-DMUSA_ARCHITECTURES=31`."
        )
    endif()

    infiniccl_normalize_musa_architectures(
        "${_detected_architectures}"
        _detected_architectures
    )
    set(${output} "${_detected_architectures}" PARENT_SCOPE)
endfunction()
