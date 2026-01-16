add_requires("pybind11")

local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")

set_toolchains("gcc")

-- Add spdlog from third_party directory
add_includedirs("third_party/spdlog/include")

target("infinicore_infer")
    set_kind("shared")
    set_symbols("debug")

    add_includedirs("include", { public = false })
    add_includedirs(INFINI_ROOT.."/include", { public = true })

    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infiniop", "infinirt", "infiniccl")
    if not is_plat("windows") then
        -- Allow loading libinfinicore_infer.so via absolute path (e.g. ctypes.CDLL)
        -- without requiring LD_LIBRARY_PATH for transitive deps in the same directory.
        add_shflags("-Wl,-rpath,$ORIGIN", {force = true})
    end

    set_languages("cxx17")
    set_warnings("all", "error")

    add_files("src/models/*.cpp")
    add_files("src/models/*/*.cpp")
    add_files("src/tensor/*.cpp")
    add_files("src/allocator/*.cpp")
    add_files("src/dataloader/*.cpp")
    add_files("src/cache_manager/*.cpp")
    add_includedirs("include")

    set_installdir(INFINI_ROOT)
    add_installfiles("include/infinicore_infer.h", {prefixdir = "include"})
    add_installfiles("include/infinicore_infer/cache.h", {prefixdir = "include/infinicore_infer"})
    add_installfiles("include/infinicore_infer/kv_compression.h", {prefixdir = "include/infinicore_infer"})
    add_installfiles("include/infinicore_infer/weights_loader.h", {prefixdir = "include/infinicore_infer"})
    add_installfiles("include/infinicore_infer/models/*.h", {prefixdir = "include/infinicore_infer/models"})
target_end()

target("_infinilm")
    add_packages("pybind11")
    set_default(false)
    add_rules("python.module", {soabi = true})
    set_languages("cxx17")
    set_kind("shared")

    local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")

    -- add_includedirs("csrc", { public = false })
    -- add_includedirs("csrc/pybind11", { public = false })
    add_includedirs(INFINI_ROOT.."/include", { public = true })
    add_includedirs("include", { public = false })
    -- spdlog is already included globally via add_includedirs at the top

    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infinicore_cpp_api", "infiniop", "infinirt", "infiniccl")

    -- Add src files
    add_files("csrc/**.cpp")
    add_files("csrc/**.cc")

    set_installdir("python/infinilm")
target_end()
