if(INFINI_RT_BUILD_TESTING)

function(add_infini_rt_test target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE infinirt)
    add_test(NAME ${target} COMMAND ${target})
endfunction()

function(add_infini_rt_backend_runtime_test backend device_type runtime_header
         expect_async_memcpy_success supports_host_memory supports_async_memory
         supports_mem_get_info supports_memset_async supports_stream_wait_event
         supports_event supports_event_elapsed_time)
    string(TOLOWER "${backend}" backend_lower)
    set(target "test_${backend_lower}_runtime")
    add_infini_rt_test(${target} test_native_runtime.cc)
    target_compile_definitions(${target}
        PRIVATE
            "INFINI_RT_TEST_BACKEND_NAME=\"${backend}\""
            "INFINI_RT_TEST_DEVICE_TYPE=${device_type}"
            "INFINI_RT_TEST_RUNTIME_HEADER=\"${runtime_header}\""
            "INFINI_RT_TEST_EXPECT_ASYNC_MEMCPY_SUCCESS=${expect_async_memcpy_success}"
            "INFINI_RT_TEST_SUPPORTS_HOST_MEMORY=${supports_host_memory}"
            "INFINI_RT_TEST_SUPPORTS_ASYNC_MEMORY=${supports_async_memory}"
            "INFINI_RT_TEST_SUPPORTS_MEM_GET_INFO=${supports_mem_get_info}"
            "INFINI_RT_TEST_SUPPORTS_MEMSET_ASYNC=${supports_memset_async}"
            "INFINI_RT_TEST_SUPPORTS_STREAM_WAIT_EVENT=${supports_stream_wait_event}"
            "INFINI_RT_TEST_SUPPORTS_EVENT=${supports_event}"
            "INFINI_RT_TEST_SUPPORTS_EVENT_ELAPSED_TIME=${supports_event_elapsed_time}")
endfunction()

function(add_infini_rt_backend_graph_test backend device_type supports_graph_capture)
    string(TOLOWER "${backend}" backend_lower)
    set(target "test_${backend_lower}_graph")
    add_infini_rt_test(${target} test_native_graph.cc)
    target_compile_definitions(${target}
        PRIVATE
            "INFINI_RT_TEST_BACKEND_NAME=\"${backend}\""
            "INFINI_RT_TEST_DEVICE_TYPE=${device_type}"
            "INFINI_RT_TEST_SUPPORTS_GRAPH_CAPTURE=${supports_graph_capture}")
endfunction()

add_infini_rt_test(test_smoke test_smoke.cc)
add_infini_rt_test(test_core test_core.cc)
add_infini_rt_test(test_small_vector test_small_vector.cc)
add_infini_rt_test(test_metadata_view test_metadata_view.cc)
add_infini_rt_test(test_shape_strides_storage test_shape_strides_storage.cc)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_infini_rt_test(test_tensor_view_allocations
                       test_tensor_view_allocations.cc)
endif()

set(INFINI_RT_TEST_HAS_RUNTIME_BACKEND OFF)

if(WITH_CPU)
    set(INFINI_RT_TEST_HAS_RUNTIME_BACKEND ON)
    add_infini_rt_backend_runtime_test(
        CPU infini::rt::Device::Type::kCpu infini/rt/cpu/runtime_.h
        0 1 0 1 0 1 1 1)
endif()

if(WITH_NVIDIA)
    set(INFINI_RT_TEST_HAS_RUNTIME_BACKEND ON)
    add_infini_rt_backend_runtime_test(
        NVIDIA infini::rt::Device::Type::kNvidia
        infini/rt/nvidia/runtime_.h
        1 1 1 1 1 1 1 1)
    add_infini_rt_backend_graph_test(
        NVIDIA infini::rt::Device::Type::kNvidia 1)
endif()

if(WITH_ILUVATAR)
    set(INFINI_RT_TEST_HAS_RUNTIME_BACKEND ON)
    add_infini_rt_backend_runtime_test(
        ILUVATAR infini::rt::Device::Type::kIluvatar
        infini/rt/iluvatar/runtime_.h
        1 1 1 1 1 1 1 1)
    add_infini_rt_backend_graph_test(
        ILUVATAR infini::rt::Device::Type::kIluvatar 1)
endif()

if(WITH_HYGON)
    set(INFINI_RT_TEST_HAS_RUNTIME_BACKEND ON)
    add_infini_rt_backend_runtime_test(
        HYGON infini::rt::Device::Type::kHygon
        infini/rt/hygon/runtime_.h
        1 1 1 1 1 1 1 1)
    add_infini_rt_backend_graph_test(
        HYGON infini::rt::Device::Type::kHygon 1)
endif()

if(WITH_THEAD)
    set(INFINI_RT_TEST_HAS_RUNTIME_BACKEND ON)
    add_infini_rt_backend_runtime_test(
        THEAD infini::rt::Device::Type::kThead
        infini/rt/thead/runtime_.h
        1 1 1 1 1 1 1 1)
    add_infini_rt_backend_graph_test(
        THEAD infini::rt::Device::Type::kThead 1)
endif()

if(WITH_METAX)
    set(INFINI_RT_TEST_HAS_RUNTIME_BACKEND ON)
    add_infini_rt_backend_runtime_test(
        METAX infini::rt::Device::Type::kMetax
        infini/rt/metax/runtime_.h
        1 1 1 1 1 1 1 1)
    add_infini_rt_backend_graph_test(
        METAX infini::rt::Device::Type::kMetax 1)
endif()

if(WITH_MARS)
    set(INFINI_RT_TEST_HAS_RUNTIME_BACKEND ON)
    add_infini_rt_backend_runtime_test(
        MARS infini::rt::Device::Type::kMars
        infini/rt/mars/runtime_.h
        1 1 1 1 1 1 1 1)
    add_infini_rt_backend_graph_test(
        MARS infini::rt::Device::Type::kMars 1)
endif()

if(WITH_MOORE)
    set(INFINI_RT_TEST_HAS_RUNTIME_BACKEND ON)
    add_infini_rt_backend_runtime_test(
        MOORE infini::rt::Device::Type::kMoore
        infini/rt/moore/runtime_.h
        1 1 0 1 1 1 1 1)
    add_infini_rt_backend_graph_test(
        MOORE infini::rt::Device::Type::kMoore 1)
endif()

if(WITH_CAMBRICON)
    set(INFINI_RT_TEST_HAS_RUNTIME_BACKEND ON)
    add_infini_rt_backend_runtime_test(
        CAMBRICON infini::rt::Device::Type::kCambricon
        infini/rt/cambricon/runtime_.h
        1 0 0 0 1 0 0 0)
    add_infini_rt_backend_graph_test(
        CAMBRICON infini::rt::Device::Type::kCambricon 1)
endif()

if(WITH_ASCEND)
    set(INFINI_RT_TEST_HAS_RUNTIME_BACKEND ON)
    add_infini_rt_backend_runtime_test(
        ASCEND infini::rt::Device::Type::kAscend
        infini/rt/ascend/runtime_.h
        1 0 0 0 1 0 0 0)
    add_infini_rt_backend_graph_test(
        ASCEND infini::rt::Device::Type::kAscend 1)
endif()

if(INFINI_RT_TEST_HAS_RUNTIME_BACKEND)
    add_infini_rt_test(test_runtime_dispatch test_runtime_dispatch.cc)

    if(WITH_CPU)
        target_compile_definitions(test_runtime_dispatch
                                   PRIVATE INFINI_RT_TEST_WITH_CPU=1)
    endif()
    if(WITH_NVIDIA)
        target_compile_definitions(test_runtime_dispatch
                                   PRIVATE INFINI_RT_TEST_WITH_NVIDIA=1)
    endif()
    if(WITH_ILUVATAR)
        target_compile_definitions(test_runtime_dispatch
                                   PRIVATE INFINI_RT_TEST_WITH_ILUVATAR=1)
    endif()
    if(WITH_HYGON)
        target_compile_definitions(test_runtime_dispatch
                                   PRIVATE INFINI_RT_TEST_WITH_HYGON=1)
    endif()
    if(WITH_THEAD)
        target_compile_definitions(test_runtime_dispatch
                                   PRIVATE INFINI_RT_TEST_WITH_THEAD=1)
    endif()
    if(WITH_METAX)
        target_compile_definitions(test_runtime_dispatch
                                   PRIVATE INFINI_RT_TEST_WITH_METAX=1)
    endif()
    if(WITH_MARS)
        target_compile_definitions(test_runtime_dispatch
                                   PRIVATE INFINI_RT_TEST_WITH_MARS=1)
    endif()
    if(WITH_MOORE)
        target_compile_definitions(test_runtime_dispatch
                                   PRIVATE INFINI_RT_TEST_WITH_MOORE=1)
    endif()
    if(WITH_CAMBRICON)
        target_compile_definitions(test_runtime_dispatch
                                   PRIVATE INFINI_RT_TEST_WITH_CAMBRICON=1)
    endif()
    if(WITH_ASCEND)
        target_compile_definitions(test_runtime_dispatch
                                   PRIVATE INFINI_RT_TEST_WITH_ASCEND=1)
    endif()
endif()

set(INFINI_RT_TEST_INSTALL_PREFIX
    "${CMAKE_CURRENT_BINARY_DIR}/install_consumer_prefix")
set(INFINI_RT_TEST_CONSUMER_BINARY
    "${CMAKE_CURRENT_BINARY_DIR}/install_consumer_build")
string(JOIN "," INFINI_RT_TEST_EXPECTED_BACKENDS
       ${INFINI_RT_PUBLIC_HEADER_DEVICES})
set(INFINI_RT_TEST_CONSUMER_BACKEND NONE)
set(INFINI_RT_TEST_CUDATOOLKIT_ROOT "${CUDAToolkit_ROOT}")
set(INFINI_RT_TEST_BACKEND_ROOT_VARIABLE "")
set(INFINI_RT_TEST_BACKEND_ROOT "")

if(NOT INFINI_RT_TEST_CUDATOOLKIT_ROOT AND CUDAToolkit_ROOT_DIR)
    set(INFINI_RT_TEST_CUDATOOLKIT_ROOT "${CUDAToolkit_ROOT_DIR}")
endif()

if(WITH_NVIDIA)
    set(INFINI_RT_TEST_CONSUMER_BACKEND NVIDIA)
elseif(WITH_ILUVATAR)
    set(INFINI_RT_TEST_CONSUMER_BACKEND ILUVATAR)
elseif(WITH_HYGON)
    set(INFINI_RT_TEST_CONSUMER_BACKEND HYGON)
    if(NOT INFINI_RT_TEST_CUDATOOLKIT_ROOT)
        set(INFINI_RT_TEST_CUDATOOLKIT_ROOT "${HYGON_CUDA_ROOT}")
    endif()
elseif(WITH_THEAD)
    set(INFINI_RT_TEST_CONSUMER_BACKEND THEAD)
    if(NOT INFINI_RT_TEST_CUDATOOLKIT_ROOT)
        set(INFINI_RT_TEST_CUDATOOLKIT_ROOT "${THEAD_CUDA_ROOT}")
    endif()
elseif(WITH_METAX)
    set(INFINI_RT_TEST_CONSUMER_BACKEND METAX)
    set(INFINI_RT_TEST_BACKEND_ROOT_VARIABLE InfiniRT_METAX_ROOT)
    set(INFINI_RT_TEST_BACKEND_ROOT "${MACA_PATH}")
elseif(WITH_MARS)
    set(INFINI_RT_TEST_CONSUMER_BACKEND MARS)
    set(INFINI_RT_TEST_BACKEND_ROOT_VARIABLE InfiniRT_MARS_ROOT)
    set(INFINI_RT_TEST_BACKEND_ROOT "${HPCC_PATH}")
elseif(WITH_MOORE)
    set(INFINI_RT_TEST_CONSUMER_BACKEND MOORE)
    set(INFINI_RT_TEST_BACKEND_ROOT_VARIABLE InfiniRT_MOORE_ROOT)
    set(INFINI_RT_TEST_BACKEND_ROOT "${MUSA_ROOT}")
elseif(WITH_CAMBRICON)
    set(INFINI_RT_TEST_CONSUMER_BACKEND CAMBRICON)
    set(INFINI_RT_TEST_BACKEND_ROOT_VARIABLE InfiniRT_CAMBRICON_ROOT)
    set(INFINI_RT_TEST_BACKEND_ROOT "${NEUWARE_HOME}")
elseif(WITH_ASCEND)
    set(INFINI_RT_TEST_CONSUMER_BACKEND ASCEND)
    set(INFINI_RT_TEST_BACKEND_ROOT_VARIABLE InfiniRT_ASCEND_ROOT)
    set(INFINI_RT_TEST_BACKEND_ROOT "${ASCEND_HOME}")
elseif(WITH_CPU)
    set(INFINI_RT_TEST_CONSUMER_BACKEND CPU)
endif()

set(INFINI_RT_TEST_INSTALL_COMMAND
    "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}"
    --prefix "${INFINI_RT_TEST_INSTALL_PREFIX}")
if(CMAKE_CONFIGURATION_TYPES)
    list(APPEND INFINI_RT_TEST_INSTALL_COMMAND --config "$<CONFIG>")
endif()
add_test(NAME test_install COMMAND ${INFINI_RT_TEST_INSTALL_COMMAND})
set_tests_properties(test_install PROPERTIES FIXTURES_SETUP infini_rt_install)

add_test(
    NAME test_install_consumer
    COMMAND "${CMAKE_COMMAND}"
            "-DINFINI_RT_INSTALL_PREFIX=${INFINI_RT_TEST_INSTALL_PREFIX}"
            "-DINFINI_RT_REQUIRED_VERSION=${PROJECT_VERSION}"
            "-DINFINI_RT_EXPECTED_BACKENDS=${INFINI_RT_TEST_EXPECTED_BACKENDS}"
            "-DINFINI_RT_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/install_consumer"
            "-DINFINI_RT_CONSUMER_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/install_consumer_smoke.cc"
            "-DINFINI_RT_CONSUMER_BINARY=${INFINI_RT_TEST_CONSUMER_BINARY}"
            "-DINFINI_RT_CMAKE_GENERATOR=${CMAKE_GENERATOR}"
            "-DINFINI_RT_CMAKE_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
            "-DINFINI_RT_CMAKE_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
            "-DINFINI_RT_CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
            "-DINFINI_RT_BUILD_CONFIG=$<CONFIG>"
            "-DINFINI_RT_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
            "-DINFINI_RT_CMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
            "-DINFINI_RT_CTEST_COMMAND=${CMAKE_CTEST_COMMAND}"
            "-DINFINI_RT_CUDATOOLKIT_ROOT=${INFINI_RT_TEST_CUDATOOLKIT_ROOT}"
            "-DINFINI_RT_BACKEND_ROOT_VARIABLE=${INFINI_RT_TEST_BACKEND_ROOT_VARIABLE}"
            "-DINFINI_RT_BACKEND_ROOT=${INFINI_RT_TEST_BACKEND_ROOT}"
            "-DINFINI_RT_CONSUMER_BACKEND=${INFINI_RT_TEST_CONSUMER_BACKEND}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/compile_install_consumer.cmake")
set_tests_properties(test_install_consumer PROPERTIES
    FIXTURES_REQUIRED infini_rt_install)

endif()

if(INFINI_RT_BUILD_PERFORMANCE_TESTING)
    add_subdirectory(performance)
endif()
