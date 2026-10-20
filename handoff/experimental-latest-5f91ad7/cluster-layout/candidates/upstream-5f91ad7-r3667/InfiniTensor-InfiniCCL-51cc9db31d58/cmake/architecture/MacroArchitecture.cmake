include_guard(GLOBAL)

# Normalize an architecture list into unique numeric architecture tokens.
# For example, with strip prefix `arch_`, `arch_31, 3.1;31` becomes `31`.
function(infiniccl_normalize_architectures input output strip_prefix description)
    set(_architectures "${input}")
    string(REPLACE "," ";" _architectures "${_architectures}")
    string(REGEX REPLACE "[ \t\r\n]+" ";" _architectures "${_architectures}")

    if("${description}" STREQUAL "")
        set(_description "architecture")
    else()
        set(_description "${description}")
    endif()

    set(_normalized)
    foreach(_architecture IN LISTS _architectures)
        if(_architecture STREQUAL "")
            continue()
        endif()

        set(_raw_architecture "${_architecture}")
        if(NOT "${strip_prefix}" STREQUAL "")
            string(REGEX REPLACE "^${strip_prefix}" "" _architecture "${_architecture}")
        endif()
        string(REPLACE "." "" _architecture "${_architecture}")

        if(NOT _architecture MATCHES "^[0-9]+$")
            if("${strip_prefix}" STREQUAL "")
                set(_accepted_values "numeric values")
            else()
                set(_accepted_values "numeric values or `${strip_prefix}<value>`")
            endif()
            message(FATAL_ERROR
                "Invalid ${_description} `${_raw_architecture}`. "
                "Use ${_accepted_values}."
            )
        endif()

        list(APPEND _normalized "${_architecture}")
    endforeach()

    list(REMOVE_DUPLICATES _normalized)
    if(NOT _normalized)
        message(FATAL_ERROR "At least one ${_description} is required.")
    endif()

    set(${output} "${_normalized}" PARENT_SCOPE)
endfunction()

# Compute one conservative SDK macro value from a normalized architecture list.
# This is for SDK headers that expose one macro-controlled API surface;
# the caller supplies the encoding scale, so `31;22` with scale `10` yields `220`.
function(infiniccl_compute_least_capable_architecture_macro architectures scale output)
    if("${scale}" STREQUAL "")
        message(FATAL_ERROR "Architecture macro scale is required.")
    endif()

    set(_macro_value)
    foreach(_architecture IN LISTS architectures)
        math(EXPR _architecture_macro_value "${_architecture} * ${scale}")
        if(NOT _macro_value OR _architecture_macro_value LESS _macro_value)
            set(_macro_value "${_architecture_macro_value}")
        endif()
    endforeach()

    if(NOT _macro_value)
        message(FATAL_ERROR "At least one architecture is required to compute a macro value.")
    endif()

    set(${output} "${_macro_value}" PARENT_SCOPE)
endfunction()

# Convert normalized architectures to compiler options using a vendor prefix,
# for example `31;22` plus `--arch=` yields `--arch=31` and `--arch=22`.
function(infiniccl_make_architecture_compile_options architectures option_prefix output)
    set(_compile_options)
    foreach(_architecture IN LISTS architectures)
        list(APPEND _compile_options "${option_prefix}${_architecture}")
    endforeach()

    set(${output} "${_compile_options}" PARENT_SCOPE)
endfunction()

# Apply an SDK-visible architecture macro and matching compiler options to one
# target only, avoiding global CMake flags that can leak into other platforms.
function(infiniccl_configure_macro_architecture_target target macro_name macro_value compile_options)
    if(NOT TARGET ${target})
        message(FATAL_ERROR "Cannot configure unknown target `${target}`.")
    endif()
    if("${macro_name}" STREQUAL "")
        message(FATAL_ERROR "Architecture macro name is required for target `${target}`.")
    endif()
    if("${macro_value}" STREQUAL "")
        message(FATAL_ERROR "Architecture macro value is required for target `${target}`.")
    endif()

    foreach(_compile_option IN LISTS compile_options)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:${_compile_option}>
        )
    endforeach()
    target_compile_definitions(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${macro_name}=${macro_value}>
    )
endfunction()
