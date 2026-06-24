cmake_minimum_required(VERSION 3.18)
project(infinilm LANGUAGES CXX)

# ==================== Basic Configuration ====================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# INFINI_ROOT environment variable
if(DEFINED ENV{INFINI_ROOT})
    set(INFINI_ROOT "$ENV{INFINI_ROOT}")
elseif(WIN32)
    set(INFINI_ROOT "$ENV{HOMEPATH}/.infini")
else()
    set(INFINI_ROOT "$ENV{HOME}/.infini")
endif()

# ==================== Compilation Options ====================
option(USE_KV_CACHING "Whether to compile the path using the kv caching operator" OFF)
if(USE_KV_CACHING)
    add_definitions(-DENABLE_KV_CACHING)
endif()

option(USE_CLASSIC_LLAMA "Whether to use the classic LlamaForCausalLM" OFF)
if(USE_CLASSIC_LLAMA)
    add_definitions(-DUSE_CLASSIC_LLAMA)
endif()

# ==================== Third-party Headers ====================
# spdlog (third_party)
include_directories("${CMAKE_SOURCE_DIR}/third_party/spdlog/include")

# json (third_party)
include_directories("${CMAKE_SOURCE_DIR}/third_party/json/single_include/")

# Infini headers
include_directories("${INFINI_ROOT}/include")

# Local includes
include_directories("${CMAKE_SOURCE_DIR}/include")

# ==================== Find pybind11 ====================
# Method 1: Use find_package
find_package(pybind11 REQUIRED)

# Method 2: If pybind11 exists as a subdirectory
# add_subdirectory(third_party/pybind11)
# Note: If using Method 1, ensure pybind11 is installed or CMAKE_PREFIX_PATH is set

# ==================== Library Link Directories ====================
link_directories("${INFINI_ROOT}/lib")

# ==================== Build Target ====================
pybind11_add_module(_infinilm MODULE
    # Source files - recursively add all files under csrc
    # Alternatively, use file(GLOB) to collect files
)

# Collect source files
file(GLOB_RECURSE INFINILM_SOURCES
    "${CMAKE_SOURCE_DIR}/csrc/*.cpp"
    "${CMAKE_SOURCE_DIR}/csrc/*.cc"
)

# Add source files to target
target_sources(_infinilm PRIVATE ${INFINILM_SOURCES})

# ==================== Link Libraries ====================
target_link_libraries(_infinilm PRIVATE
    infinicore_cpp_api
    infiniop
    infinirt
    infiniccl
)

# ==================== Target Properties ====================
# Set output directory (corresponds to xmake's set_installdir)
set_target_properties(_infinilm PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/infinilm/lib"
    PREFIX ""  # Remove lib prefix
)

# If pybind11 requires specific compile definitions
target_compile_definitions(_infinilm PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
)

# ==================== Python Module Installation (Optional) ====================
# install(TARGETS _infinilm
#     LIBRARY DESTINATION python/infinilm/lib
# )
