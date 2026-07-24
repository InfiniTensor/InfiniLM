add_requires("pybind11")

local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")
local CUDA_ROOT = os.getenv("CUDA_HOME") or os.getenv("CUDA_PATH") or "/usr/local/cuda"

set_toolchains("gcc")

add_includedirs("third_party/spdlog/include")
add_includedirs("third_party/json/single_include/")
if os.isdir(CUDA_ROOT .. "/include") then
    add_includedirs(CUDA_ROOT .. "/include")
end

target("infinicore_runtime")
    set_default(false)
    set_kind("shared")
    set_languages("cxx17")
    add_defines("ENABLE_INFINIOPS_API", "USE_INFINIRT_GRAPH")

    if not is_plat("windows") then
        add_cxflags("-fPIC")
    end

    add_includedirs("csrc/infinicore/include", { public = true })
    add_includedirs("csrc/infinicore/src")
    add_includedirs("csrc/infinicore/utils")
    add_includedirs(INFINI_ROOT .. "/include", { public = true })
    add_includedirs(INFINI_ROOT .. "/include/infiniccl", { public = true })

    add_linkdirs(INFINI_ROOT .. "/lib", INFINI_ROOT .. "/lib64", { public = true })
    if is_plat("linux") then
        add_rpathdirs("$ORIGIN")
    elseif is_plat("macosx") then
        add_rpathdirs("@loader_path")
    end
    add_links("infiniops", "infiniccl", "infinirt", { public = true })

    add_files("csrc/infinicore/src/*.cc")
    add_files("csrc/infinicore/src/context/*.cc")
    add_files("csrc/infinicore/src/context/*/*.cc")
    add_files("csrc/infinicore/src/graph/*.cc")
    add_files("csrc/infinicore/src/nn/*.cc")
    add_files("csrc/infinicore/src/ops/*/*.cc")
    add_files("csrc/infinicore/src/ops/dequant/*/*.cc")
    add_files("csrc/infinicore/src/ops/quant/*/*.cc")
    add_files("csrc/infinicore/src/tensor/*.cc")
    add_files("csrc/infinicore/utils/*.cc")

    remove_files("csrc/infinicore/src/ops/*/*_cpu.cc")
    remove_files("csrc/infinicore/src/ops/*/*_flashattn.cc")
    remove_files("csrc/infinicore/src/ops/*/*_hygon.cc")
    remove_files("csrc/infinicore/src/ops/*/*_moore.cc")

    set_installdir("python/infinicore")
target_end()

target("_infinicore")
    add_packages("pybind11")
    set_default(false)
    add_rules("python.module", { soabi = true })
    set_languages("cxx17")
    set_kind("shared")

    add_deps("infinicore_runtime")

    if is_plat("linux") then
        add_rpathdirs("$ORIGIN")
    elseif is_plat("macosx") then
        add_rpathdirs("@loader_path")
    end

    add_includedirs("csrc/infinicore/include")
    add_includedirs("csrc/infinicore/src")
    add_includedirs("csrc/infinicore/utils")
    add_includedirs(INFINI_ROOT .. "/include")

    add_files("csrc/infinicore/src/pybind11/infinicore.cc")
    add_files("csrc/infinicore/src/pybind11/from_list.cc")

    set_installdir("python/infinicore")
target_end()

target("_infinilm")
    add_packages("pybind11")
    set_default(false)
    add_rules("python.module", { soabi = true })
    set_languages("cxx17")
    set_kind("shared")

    add_deps("infinicore_runtime")

    if is_plat("linux") then
        add_rpathdirs("$ORIGIN/../../infinicore/lib")
    elseif is_plat("macosx") then
        add_rpathdirs("@loader_path/../../infinicore/lib")
    end

    add_includedirs("csrc/infinicore/include")
    add_includedirs(INFINI_ROOT .. "/include")
    add_includedirs(INFINI_ROOT .. "/include/infiniccl")

    add_files("csrc/**.cpp")
    add_files("csrc/**.cc")
    remove_files("csrc/infinicore/**.cpp")
    remove_files("csrc/infinicore/**.cc")

    set_installdir("python/infinilm")
target_end()
