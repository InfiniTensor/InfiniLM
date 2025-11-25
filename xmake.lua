add_requires("pybind11")

local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")

set_toolchains("gcc")

-- Add spdlog from third_party directory
add_includedirs("third_party/spdlog/include")

target("infinicore_infer")
    set_kind("shared")

    add_includedirs("include", { public = false })
    add_includedirs(INFINI_ROOT.."/include", { public = true })

    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infiniop", "infinirt", "infiniccl")

    set_languages("cxx17")
    set_warnings("all", "error")

    add_files("src/models/*.cpp")
    add_files("src/models/*/*.cpp")
    add_files("csrc/models/llama/*.cpp")
    add_files("src/tensor/*.cpp")
    add_files("src/allocator/*.cpp")
    add_files("src/dataloader/*.cpp")
    add_files("src/cache_manager/*.cpp")
    add_includedirs("include")

    set_installdir(INFINI_ROOT)
    add_installfiles("include/infinicore_infer.h", {prefixdir = "include"})
    add_installfiles("include/infinicore_infer/models/*.h", {prefixdir = "include/infinicore_infer/models"})
target_end()
-- Test target for Llama model skeleton
target("test_llama_skeleton")
    set_kind("binary")
    set_default(false)

    set_languages("cxx17")
    set_warnings("all", "error")

    local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")

    add_includedirs("csrc", { public = false })
    add_includedirs("include", { public = false })
    add_includedirs(INFINI_ROOT.."/include", { public = true })

    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infinicore_c_api", "infiniop", "infinirt", "infiniccl")

    -- Add Llama model files (exclude test file from wildcard, add it separately)
    add_files("csrc/models/llama/llama_*.cpp")
    add_files("csrc/models/llama/test_llama_skeleton.cpp")

    set_installdir(INFINI_ROOT)
target_end()

-- Python bindings for Llama model
target("_infinilm_llama")
    add_packages("pybind11")
    set_default(false)
    add_rules("python.module", {soabi = true})
    set_languages("cxx17")
    set_kind("shared")

    local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")

    add_includedirs("csrc", { public = false })
    add_includedirs("include", { public = false })
    add_includedirs(INFINI_ROOT.."/include", { public = true })
    -- spdlog is already included globally via add_includedirs at the top

    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infinicore_c_api", "infiniop", "infinirt", "infiniccl")

    -- Add Llama model files
    add_files("csrc/models/llama/llama_*.cpp")
    add_files("csrc/models/llama/pybind11_module.cc")

    set_installdir("python/infinilm")
target_end()
