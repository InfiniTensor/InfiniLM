add_requires("pybind11")

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

local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")

target("_infinilm")
    add_packages("pybind11")
    set_default(false)
    add_rules("python.module", {soabi = true})
    set_languages("cxx17")
    set_kind("shared")

    -- add_includedirs("csrc", { public = false })
    -- add_includedirs("csrc/pybind11", { public = false })
    add_includedirs(INFINI_ROOT.."/include", { public = true })
    -- spdlog is already included globally via add_includedirs at the top

    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infinicore_cpp_api", "infiniop", "infinirt", "infiniccl")

    -- Add C++ sources
    add_files("csrc/**.cpp")
    add_files("csrc/**.cc")

    set_installdir("python/infinilm")
target_end()

target("compressed_tensors_config_test")
    set_default(false)
    set_kind("binary")
    set_languages("cxx17")

    add_includedirs(".")
    add_files("test/config/compressed_tensors_config_test.cpp")
    add_files("csrc/config/compressed_tensors_config.cpp")
    add_files("csrc/config/module_target_matcher.cpp")
target_end()

target("quant_config_test")
    set_default(false)
    set_kind("binary")
    set_languages("cxx17")

    add_includedirs(".")
    add_includedirs(INFINI_ROOT.."/include")
    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infinicore_cpp_api", "infiniop", "infinirt", "infiniccl")
    add_runenvs("LD_LIBRARY_PATH", INFINI_ROOT.."/lib", {pathenv = true})
    add_files("test/config/quant_config_test.cpp")
    add_files("csrc/config/compressed_tensors_config.cpp")
    add_files("csrc/config/module_target_matcher.cpp")
    add_files("csrc/config/quant_config.cpp")
    add_files("csrc/global_state/global_state.cpp")
    add_files("csrc/layers/quantization/base_quantization.cpp")
    add_files("csrc/layers/quantization/none_quantization.cpp")
    add_files("csrc/layers/quantization/compressed_tensors.cpp")
    add_files("csrc/layers/quantization/awq.cpp")
    add_files("csrc/layers/quantization/awq_marlin.cpp")
    add_files("csrc/layers/quantization/gptq.cpp")
    add_files("csrc/layers/quantization/gptq_marlin.cpp")
    add_files("csrc/layers/quantization/gptq_qy.cpp")
    add_files("csrc/layers/quantization/marlin_utils.cpp")
    add_files("csrc/layers/quantization/mxfp4.cpp")
target_end()

target("module_target_matcher_test")
    set_default(false)
    set_kind("binary")
    set_languages("cxx17")

    add_includedirs(".")
    add_files("test/config/module_target_matcher_test.cpp")
    add_files("csrc/config/module_target_matcher.cpp")
target_end()
