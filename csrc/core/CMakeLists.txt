file(GLOB_RECURSE INFINILM_CORE_SRC CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
)
list(FILTER INFINILM_CORE_SRC EXCLUDE REGEX ".*/pybind11/.*")

add_library(infinilm_core STATIC ${INFINILM_CORE_SRC})

target_include_directories(infinilm_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/../.."
    "${CMAKE_CURRENT_SOURCE_DIR}/../../csrc"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/spdlog/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/json/single_include"
)

target_link_libraries(infinilm_core PUBLIC infinirt infiniccl)

target_compile_options(infinilm_core PRIVATE
    -Wno-deprecated-declarations
    -Wno-unused-function
)

if(INFINILM_ENABLE_ASCEND)
    target_compile_definitions(infinilm_core PUBLIC ENABLE_ASCEND_API)
else()
    target_compile_definitions(infinilm_core PUBLIC "ENABLE_${INFINI_NV_BACKEND}_API")
endif()
