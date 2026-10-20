cmake_minimum_required(VERSION 3.18)
project(infinirt_install_consumer LANGUAGES CXX)

if(NOT INFINI_RT_CONSUMER_SOURCE)
    message(FATAL_ERROR "INFINI_RT_CONSUMER_SOURCE is required.")
endif()
if(NOT INFINI_RT_REQUIRED_VERSION)
    message(FATAL_ERROR "INFINI_RT_REQUIRED_VERSION is required.")
endif()
if(NOT INFINI_RT_EXPECTED_BACKENDS)
    message(FATAL_ERROR "INFINI_RT_EXPECTED_BACKENDS is required.")
endif()

find_package(InfiniRT ${INFINI_RT_REQUIRED_VERSION} EXACT CONFIG REQUIRED)

if(NOT TARGET InfiniRT::infinirt)
    message(FATAL_ERROR "The InfiniRT::infinirt target was not imported.")
endif()
string(REPLACE "," ";" _infinirt_expected_backends
       "${INFINI_RT_EXPECTED_BACKENDS}")
if(NOT "${InfiniRT_ENABLED_BACKENDS}" STREQUAL
       "${_infinirt_expected_backends}")
    message(FATAL_ERROR
            "InfiniRT reported enabled backends '${InfiniRT_ENABLED_BACKENDS}', expected '${_infinirt_expected_backends}'.")
endif()

foreach(_infinirt_backend CPU NVIDIA ILUVATAR HYGON THEAD METAX MARS MOORE CAMBRICON ASCEND)
    string(TOLOWER "${_infinirt_backend}" _infinirt_backend_lower)
    list(FIND _infinirt_expected_backends "${_infinirt_backend_lower}"
         _infinirt_backend_index)
    if(_infinirt_backend_index EQUAL -1)
        set(_infinirt_expected_state OFF)
    else()
        set(_infinirt_expected_state ON)
    endif()

    if(NOT DEFINED InfiniRT_WITH_${_infinirt_backend})
        message(FATAL_ERROR
                "InfiniRT did not report InfiniRT_WITH_${_infinirt_backend}.")
    endif()
    if(NOT "${InfiniRT_WITH_${_infinirt_backend}}" STREQUAL
           "${_infinirt_expected_state}")
        message(FATAL_ERROR
                "InfiniRT_WITH_${_infinirt_backend} is '${InfiniRT_WITH_${_infinirt_backend}}', expected '${_infinirt_expected_state}'.")
    endif()
endforeach()

if(INFINI_RT_CONSUMER_BACKEND
   AND NOT INFINI_RT_CONSUMER_BACKEND STREQUAL "NONE")
    string(TOLOWER "${INFINI_RT_CONSUMER_BACKEND}"
           _infinirt_consumer_backend_lower)
    list(FIND _infinirt_expected_backends
         "${_infinirt_consumer_backend_lower}"
         _infinirt_consumer_backend_index)
    if(_infinirt_consumer_backend_index EQUAL -1)
        message(FATAL_ERROR
                "The consumer backend '${INFINI_RT_CONSUMER_BACKEND}' is not enabled in InfiniRT.")
    endif()
endif()

add_executable(infinirt_install_consumer "${INFINI_RT_CONSUMER_SOURCE}")
target_link_libraries(infinirt_install_consumer PRIVATE InfiniRT::infinirt)

if(INFINI_RT_CONSUMER_BACKEND
   AND NOT INFINI_RT_CONSUMER_BACKEND STREQUAL "NONE")
    target_compile_definitions(
        infinirt_install_consumer
        PRIVATE
            "INFINI_RT_CONSUMER_BACKEND_${INFINI_RT_CONSUMER_BACKEND}=1")
endif()

enable_testing()
add_test(NAME run_infinirt_install_consumer
         COMMAND infinirt_install_consumer)
