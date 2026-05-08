cmake_minimum_required(VERSION 3.20)
project(InfiniLM VERSION 0.1.0 LANGUAGES CXX)

# Fix pybind11 warning
cmake_policy(SET CMP0148 OLD)
set(PYBIND11_FINDPYTHON ON)

# Determine INFINI_ROOT
if(NOT DEFINED ENV{INFINI_ROOT})
    if(WIN32)
        set(INFINI_ROOT "$ENV{HOMEPATH}/.infini")
    else()
        set(INFINI_ROOT "$ENV{HOME}/.infini")
    endif()
else()
    set(INFINI_ROOT "$ENV{INFINI_ROOT}")
endif()

# Options for kv-caching and classic llama
option(USE_KV_CACHING "Whether to compile the path using the kv caching operator" OFF)
option(USE_CLASSIC_LLAMA "Whether to using the classic LlamaForCausalLM" OFF)
option(USE_WERROR "Treat warnings as errors" OFF)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find required packages
find_package(pybind11 REQUIRED)

# Add spdlog from third_party directory
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/spdlog/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/json/single_include/
)

# Add defines based on options
if(USE_KV_CACHING)
    add_compile_definitions(ENABLE_KV_CACHING)
endif()

if(USE_CLASSIC_LLAMA)
    add_compile_definitions(USE_CLASSIC_LLAMA)
endif()

# ============================================
# Target 1: infinicore_infer (shared library)
# ============================================

# Collect all source files
file(GLOB_RECURSE INFINICORE_INFER_SRC
    "src/models/*.cpp"
    "src/tensor/*.cpp"
    "src/allocator/*.cpp"
    "src/dataloader/*.cpp"
    "src/cache_manager/*.cpp"
)

if(INFINICORE_INFER_SRC)
    add_library(infinicore_infer SHARED ${INFINICORE_INFER_SRC})

    target_include_directories(infinicore_infer 
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
        PUBLIC ${INFINI_ROOT}/include
    )

    target_link_directories(infinicore_infer 
        PRIVATE ${INFINI_ROOT}/lib
    )

    target_link_libraries(infinicore_infer 
        PRIVATE infiniop infinirt infiniccl
    )

    target_compile_options(infinicore_infer PRIVATE -Wall)
    if(USE_WERROR)
        target_compile_options(infinicore_infer PRIVATE -Werror)
    endif()

    # Suppress specific warnings that are common in the codebase
    target_compile_options(infinicore_infer PRIVATE 
        -Wno-reorder
        -Wno-strict-aliasing
        -Wno-unused-variable
        -Wno-unused-but-set-variable
        -Wno-sign-compare
        -Wno-unused-local-typedefs
    )

    # Install infinicore_infer
    install(TARGETS infinicore_infer
        LIBRARY DESTINATION ${INFINI_ROOT}/lib
        ARCHIVE DESTINATION ${INFINI_ROOT}/lib
    )

    install(FILES include/infinicore_infer.h
        DESTINATION ${INFINI_ROOT}/include
    )

    file(GLOB MODEL_HEADERS include/infinicore_infer/models/*.h)
    if(MODEL_HEADERS)
        install(FILES ${MODEL_HEADERS}
            DESTINATION ${INFINI_ROOT}/include/infinicore_infer/models
        )
    endif()
endif()

# ============================================
# Target 2: _infinilm (Python module)
# ============================================

# Collect all source files for _infinilm
file(GLOB_RECURSE INFINILM_SRC
    "csrc/*.cpp"
    "csrc/*.cc"
)

# Debug output
message(STATUS "INFINILM_SRC: ${INFINILM_SRC}")
message(STATUS "Current source dir: ${CMAKE_CURRENT_SOURCE_DIR}")

if(NOT INFINILM_SRC)
    message(FATAL_ERROR "No source files found for _infinilm target! Please check if csrc/ directory exists with .cpp or .cc files.")
endif()

pybind11_add_module(_infinilm ${INFINILM_SRC})

target_include_directories(_infinilm 
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
    PUBLIC ${INFINI_ROOT}/include
)

target_link_directories(_infinilm 
    PRIVATE ${INFINI_ROOT}/lib
)

target_link_libraries(_infinilm 
    PRIVATE infinicore_cpp_api infiniop infinirt infiniccl
)

# Compiler options - use Wall but not Werror
target_compile_options(_infinilm PRIVATE -Wall)
if(USE_WERROR)
    target_compile_options(_infinilm PRIVATE -Werror)
endif()

# Suppress specific warnings that are common in the codebase
target_compile_options(_infinilm PRIVATE 
    -Wno-reorder
    -Wno-strict-aliasing
    -Wno-unused-variable
    -Wno-unused-but-set-variable
    -Wno-sign-compare
    -Wno-unused-local-typedefs
)

# Set the output directory for the Python module
set_target_properties(_infinilm PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/infinilm"
    PREFIX ""
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

# On Windows, also set the runtime output directory
if(WIN32)
    set_target_properties(_infinilm PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/infinilm"
    )
endif()

# Install _infinilm to the project's python directory
# Use FULL path with PROJECT_SOURCE_DIR to ensure correct location
install(TARGETS _infinilm
    LIBRARY DESTINATION "${CMAKE_SOURCE_DIR}/python/infinilm/lib"
)
