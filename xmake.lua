set_toolchains("gcc")

option("cxx11-abi")
    set_default(nil)
    set_showmenu(true)
    set_description("Set _GLIBCXX_USE_CXX11_ABI to match the installed InfiniCore")
    set_values("0", "1")
option_end()

local cxx11_abi = os.getenv("INFINILM_CXX11_ABI") or get_config("cxx11-abi")
if cxx11_abi and cxx11_abi ~= "" then
    if cxx11_abi ~= "0" and cxx11_abi ~= "1" then
        raise("INFINILM_CXX11_ABI must be 0 or 1")
    end
    add_defines("_GLIBCXX_USE_CXX11_ABI=" .. cxx11_abi)
end

-- Add spdlog from third_party directory
add_includedirs("third_party/spdlog/include")
add_includedirs("third_party/json/single_include/")
add_includedirs("/usr/include/python3.12")

target("_infinilm")
    set_default(false)
    -- The python.module rule is not shipped by the minimal xmake package used
    -- on the NVIDIA benchmark host. A shared library is importable by Python
    -- on Linux and keeps the build self-contained.
    set_languages("cxx17")
    set_kind("shared")
    set_prefixname("")

    local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")

    -- add_includedirs("csrc", { public = false })
    -- add_includedirs("csrc/pybind11", { public = false })
    add_includedirs(INFINI_ROOT.."/include", { public = true })
    -- spdlog is already included globally via add_includedirs at the top

    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infinicore_cpp_api", "infiniop", "infinirt", "infiniccl", "fmt")
    add_shflags("-Wl,--no-as-needed", "-lfmt", "-Wl,--as-needed", { force = true })

    -- Add C++ sources
    add_files("csrc/**.cpp")
    add_files("csrc/**.cc")

    set_installdir("python/infinilm")
target_end()
