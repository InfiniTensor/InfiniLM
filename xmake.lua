-- Prefer pip/system pybind11 to avoid xmake-repo network fetches in offline containers.
local PYBIND11_INCLUDE = os.getenv("PYBIND11_INCLUDE")
if not PYBIND11_INCLUDE or PYBIND11_INCLUDE == "" then
    local ok, out = pcall(function()
        return os.iorunv(os.getenv("PYTHON") or "python3", {
            "-c", "import pybind11; print(pybind11.get_include())"
        })
    end)
    if ok and out and out:trim() ~= "" then
        PYBIND11_INCLUDE = out:trim()
    end
end
if not PYBIND11_INCLUDE or PYBIND11_INCLUDE == "" then
    add_requires("pybind11")
    if is_plat("linux") then
        add_requireconfs("pybind11.python", { configs = { headeronly = true } })
    end
end

local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")
local CUDA_ROOT = os.getenv("CUDA_HOME") or os.getenv("CUDA_PATH") or "/usr/local/cuda"
local NEUWARE_ROOT = os.getenv("NEUWARE_HOME") or "/usr/local/neuware"
local ASCEND_ROOT = os.getenv("ASCEND_HOME_PATH") or os.getenv("ASCEND_TOOLKIT_HOME") or os.getenv("ASCEND_HOME") or "/usr/local/Ascend/ascend-toolkit/latest"
local MUSA_ROOT =
    os.getenv("MUSA_ROOT")
    or os.getenv("MUSA_HOME")
    or os.getenv("MUSA_PATH")
    or "/usr/local/musa"
local OPERATOR_CALLS_HEADER = INFINI_ROOT .. "/include/infini/operator_call_instantiations.h"

set_toolchains("gcc")

-- Fallback when xmake python plugin / network repos are unavailable.
rule("python.module")
    on_config(function (target)
        target:set("kind", "shared")
        target:set("prefixname", "")
        local suffix = try {
            function ()
                return os.iorunv(os.getenv("PYTHON") or "python3", {
                    "-c", "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '')"
                }):trim()
            end
        }
        if suffix and suffix ~= "" and suffix ~= "None" then
            target:set("extension", suffix)
        end
    end)
rule_end()

option("cxx11-abi")
    set_default(nil)
    set_showmenu(true)
    set_description("Set _GLIBCXX_USE_CXX11_ABI to match PyTorch and the installed Infini stack")
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
if os.isdir(CUDA_ROOT .. "/include") then
    add_includedirs(CUDA_ROOT .. "/include")
end
if os.isdir(NEUWARE_ROOT .. "/include") then
    add_includedirs(NEUWARE_ROOT .. "/include")
    add_linkdirs(NEUWARE_ROOT .. "/lib64")
end
if os.isdir(ASCEND_ROOT .. "/include") then
    add_includedirs(ASCEND_ROOT .. "/include")
    add_linkdirs(ASCEND_ROOT .. "/lib64")
end
if os.isdir(MUSA_ROOT .. "/include") then
    add_includedirs(MUSA_ROOT .. "/include")
    add_linkdirs(MUSA_ROOT .. "/lib", MUSA_ROOT .. "/lib64")
end

local QY_SDK_PRESENT = os.getenv("QY_ROOT") or os.getenv("QY_HOME") or os.getenv("QY_PATH")
    or os.isdir("/usr/local/denglin/sdk") or os.isdir("/usr/local/dlgpu/sdk")

target("infinicore_runtime")
    set_default(false)
    set_kind("shared")
    set_languages("cxx17")
    add_defines("ENABLE_INFINIOPS_API", "USE_INFINIRT_GRAPH")
    -- Denglin QY: CUDA-compatible streams + ATen bridge for `_dl_C` MoE.
    if QY_SDK_PRESENT then
        add_defines("ENABLE_QY_API", "ENABLE_ATEN")
    end

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
    add_files("csrc/infinicore/src/adaptor/*.cc")
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

    on_load(function (target)
        if QY_SDK_PRESENT then
            local python = os.getenv("PYTHON") or "python3"
            local torch_dir = try {
                function ()
                    return os.iorunv(python, {"-c", "import torch, os; print(os.path.dirname(torch.__file__))"}):trim()
                end
            }
            if torch_dir and torch_dir ~= "" then
                target:add("includedirs",
                    path.join(torch_dir, "include"),
                    path.join(torch_dir, "include/torch/csrc/api/include"),
                    { public = true })
                target:add("linkdirs", path.join(torch_dir, "lib"), { public = true })
                target:add("rpathdirs", path.join(torch_dir, "lib"))
                target:add("links",
                    "torch", "torch_cpu", "torch_cuda", "torch_python",
                    "c10", "c10_cuda",
                    { public = true })
            end
            local qy_root = os.getenv("QY_ROOT") or os.getenv("QY_HOME") or os.getenv("QY_PATH")
            if (not qy_root or qy_root == "") and os.isdir("/usr/local/dlgpu/sdk") then
                qy_root = "/usr/local/dlgpu"
            end
            if (not qy_root or qy_root == "") and os.isdir("/usr/local/denglin/sdk") then
                qy_root = "/usr/local/denglin"
            end
            if qy_root and os.isdir(qy_root .. "/sdk/include") then
                target:add("includedirs", qy_root .. "/sdk/include", { public = true })
                target:add("linkdirs", qy_root .. "/sdk/lib", { public = true })
                target:add("links", "curt", { public = true })
            end
        end

        if not os.isfile(OPERATOR_CALLS_HEADER) then
            return
        end
        local operator_calls = io.readfile(OPERATOR_CALLS_HEADER)
        for _, sourcefile in ipairs(os.files("csrc/infinicore/src/ops/*/*_infiniops.cc")) do
            local source = io.readfile(sourcefile)
            local missing_call = false
            for operator_name in source:gmatch('#include%s+"base/([^"]+)%.h"') do
                if not operator_calls:find('base/' .. operator_name .. '.h', 1, true) then
                    missing_call = true
                    break
                end
            end
            if not missing_call then
                for operator_type in source:gmatch("infini::ops::([%w_]+)::Call") do
                    local operator_instantiation =
                        "Operator<::infini::ops::" .. operator_type .. ">"
                    if not operator_calls:find(operator_instantiation, 1, true) then
                        missing_call = true
                        break
                    end
                end
            end
            if missing_call then
                target:remove("files", sourcefile)
            end
        end
    end)

    set_installdir("python/infinicore")
target_end()

target("_infinicore")
    if PYBIND11_INCLUDE and PYBIND11_INCLUDE ~= "" then
        add_includedirs(PYBIND11_INCLUDE)
    else
        add_packages("pybind11")
    end
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
    if PYBIND11_INCLUDE and PYBIND11_INCLUDE ~= "" then
        add_includedirs(PYBIND11_INCLUDE)
    else
        add_packages("pybind11")
    end
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

    add_files("csrc/**.cpp|infinicore/**.cpp|models/**.cpp")
    add_files("csrc/**.cc|infinicore/**.cc|models/**.cc")
    add_files("csrc/models/*.cpp")
    add_files("csrc/models/baichuan/*.cpp")
    add_files("csrc/models/chatglm/*.cpp")
    add_files("csrc/models/fm9g/*.cpp")
    add_files("csrc/models/glm4/*.cpp")
    add_files("csrc/models/internlm3/*.cpp")
    add_files("csrc/models/llama/*.cpp")
    add_files("csrc/models/minicpm4/*.cpp")
    add_files("csrc/models/minicpm5_moe/*.cpp")
    add_files("csrc/models/minicpm_eagle/*.cpp")
    add_files("csrc/models/qwen2/*.cpp")
    add_files("csrc/models/qwen3/*.cpp")

    set_installdir("python/infinilm")
target_end()
