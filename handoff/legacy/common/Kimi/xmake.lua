local PYBIND11_INCLUDE_DIR = os.getenv("PYBIND11_INCLUDE_DIR")
if PYBIND11_INCLUDE_DIR then
    if not os.isfile(PYBIND11_INCLUDE_DIR .. "/pybind11/pybind11.h") then raise("Pinned pybind11 header missing") end
    add_includedirs(PYBIND11_INCLUDE_DIR)
else
    add_requires("pybind11 3.0.4", {system = false})
end

-- Core owns SDK builds; LM consumes an explicitly installed prefix.
local INFINI_ROOT = os.getenv("INFINI_ROOT")
if not INFINI_ROOT or INFINI_ROOT == "" then
    raise("INFINI_ROOT is required. Build the pinned stack using third_party/InfiniCore/README.md, then set INFINI_ROOT to its installed prefix.")
end
local CUDA_ROOT = os.getenv("CUDA_HOME") or os.getenv("CUDA_PATH") or "/usr/local/cuda"
local NEUWARE_ROOT = os.getenv("NEUWARE_HOME") or "/usr/local/neuware"
local ASCEND_ROOT = os.getenv("ASCEND_HOME_PATH") or os.getenv("ASCEND_TOOLKIT_HOME") or os.getenv("ASCEND_HOME") or "/usr/local/Ascend/ascend-toolkit/latest"
local MUSA_ROOT =
    os.getenv("MUSA_ROOT")
    or os.getenv("MUSA_HOME")
    or os.getenv("MUSA_PATH")
    or "/usr/local/musa"
local OPERATOR_CALLS_HEADER = INFINI_ROOT .. "/include/infini/operator_call_instantiations.h"
local TORCH_LIB = os.getenv("TORCH_LIB")
set_toolchains("gcc")

-- A build configured with `-m release` does not gain optimization unless the
-- project selects a mode rule or optimization explicitly. Keep symbols for
-- runtime diagnostics while optimizing every production target.
if is_mode("release") then
    set_optimize("fastest")
end

option("cxx11-abi")
    set_default(nil)
    set_showmenu(true)
    set_description("Set _GLIBCXX_USE_CXX11_ABI to match PyTorch and the installed Infini stack")
    set_values("0", "1")
option_end()

option("capability-profile")
    set_default("release")
    set_showmenu(true)
    set_description("InfiniOps adapter capabilities: release, attention_bootstrap, or greedy_bootstrap")
    set_values("release", "attention_bootstrap", "greedy_bootstrap")
option_end()

local CAPABILITY_PROFILE = get_config("capability-profile") or "release"
-- One public-Key capability map for direct consumers and retained wrappers/adapters.
-- true requires the generated API; strings name existing explicit reject guards.
-- The three profile-dependent entries retain the original bootstrap exclusions.
local DIRECT_OPS = {
    ["SituAndMul"] = true,
    ["IndexedKimiDeltaAttention"] = true,
    ["FusedMoeMxfp4"] = true,
    ["Add"] = true,
    ["Argmax"] = true,
    ["Cast"] = true,
    ["Cat"] = true,
    ["CausalConv1d"] = true,
    ["ChunkGatedDeltaRule"] = "INFINILM_CHUNK_GATED_DELTA_RULE_OPS_UNAVAILABLE",
    ["Convolution"] = true,
    ["Copy"] = true,
    ["DeepseekMoe"] = "INFINILM_DEEPSEEK_MOE_OPS_UNAVAILABLE",
    ["Embedding"] = true,
    ["Fill"] = true,
    ["FlashAttnVarlenFunc"] = CAPABILITY_PROFILE ~= "greedy_bootstrap",
    ["FlashAttnWithKvcache"] = CAPABILITY_PROFILE ~= "greedy_bootstrap",
    ["FusedAddRmsNorm"] = true,
    ["FusedGatedDeltaNetGating"] = "INFINILM_FUSED_GATED_DELTA_NET_GATING_OPS_UNAVAILABLE",
    ["Gelu"] = true,
    ["Gemm"] = true,
    ["I8Gemm"] = true,
    ["LayerNorm"] = true,
    ["MambaSelectiveScan"] = "INFINILM_MAMBA_SELECTIVE_SCAN_OPS_UNAVAILABLE",
    ["MoeAlignBlockSize"] = "INFINICORE_MOE_ALIGN_OPS_UNAVAILABLE",
    ["MoeFusedDense"] = "INFINILM_MOE_FUSED_DENSE_OPS_UNAVAILABLE",
    ["MoeFusedGate"] = true,
    ["Mrope"] = "INFINILM_MROPE_OPS_UNAVAILABLE",
    ["Mul"] = true,
    ["MulScalar"] = "INFINICORE_MUL_SCALAR_OPS_UNAVAILABLE",
    ["NsaCompressPagedCache"] = "INFINICORE_NSA_OPS_UNAVAILABLE",
    ["NsaPagedAttention"] = "INFINICORE_NSA_OPS_UNAVAILABLE",
    ["PadConstant"] = "INFINILM_PAD_CONSTANT_OPS_UNAVAILABLE",
    ["PerChannelQuantI8"] = true,
    ["PerTensorDequantI8"] = "INFINILM_KV_INT8_OPS_UNAVAILABLE",
    ["PerTensorQuantI8"] = "INFINILM_KV_INT8_OPS_UNAVAILABLE",
    ["QuickGelu"] = "INFINICORE_QUICK_GELU_OPS_UNAVAILABLE",
    ["RecurrentGatedDeltaRule"] = "INFINILM_RECURRENT_GATED_DELTA_RULE_OPS_UNAVAILABLE",
    ["Relu"] = true,
    ["ReshapeAndCacheFlash"] = true,
    ["RmsNorm"] = true,
    ["RotaryEmbedding"] = true,
    ["SeparateTableRope"] = true,
    ["Sigmoid"] = true,
    ["Silu"] = true,
    ["SiluAndMul"] = true,
    ["Softmax"] = true,
    ["Sum"] = true,
    ["Take"] = true,
    ["TopKTopPSamplingFromLogits"] = CAPABILITY_PROFILE == "release",
    ["TopkSigmoid"] = "INFINICORE_MOE_TOPK_SIGMOID_OPS_UNAVAILABLE",
    ["TopkSoftmax"] = true,
    ["Tril"] = true,
    ["Triu"] = true,
    ["UpsampleBicubic2d"] = "INFINICORE_UPSAMPLE_BICUBIC_OPS_UNAVAILABLE",
    ["UpsampleBilinear2d"] = "INFINICORE_UPSAMPLE_OPS_UNAVAILABLE",
    ["W4MarlinGemm"] = "INFINILM_W4_MARLIN_OPS_UNAVAILABLE",
    ["W4MarlinParameters"] = "INFINILM_W4_MARLIN_OPS_UNAVAILABLE",
    ["W4MarlinRepack"] = "INFINILM_W4_MARLIN_OPS_UNAVAILABLE",
}

local cxx11_abi = os.getenv("INFINILM_CXX11_ABI")
if not cxx11_abi or cxx11_abi == "" then
    cxx11_abi = get_config("cxx11-abi")
    -- xmake stores command-line 0/1 option values as Lua booleans.
    if cxx11_abi == false then
        cxx11_abi = "0"
    elseif cxx11_abi == true then
        cxx11_abi = "1"
    end
end
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

target("infinilm_runtime")
    set_default(false)
    set_kind("shared")
    set_languages("cxx17")
    add_defines("USE_INFINIRT_GRAPH")
    if CAPABILITY_PROFILE == "greedy_bootstrap" then
        add_defines("INFINICORE_MHA_OMITTED", { public = true })
        add_defines("INFINICORE_MHA_KVCACHE_OMITTED", { public = true })
        add_defines("INFINICORE_MHA_VARLEN_OMITTED", { public = true })
    end
    if CAPABILITY_PROFILE ~= "release" then
        add_defines("INFINICORE_RANDOM_SAMPLE_OMITTED", { public = true })
    end

    if not is_plat("windows") then
        add_cxflags("-fPIC")
    end

    add_includedirs("csrc", { public = true })
    add_includedirs(INFINI_ROOT .. "/include", { public = true })
    add_includedirs(INFINI_ROOT .. "/include/infiniccl", { public = true })

    add_linkdirs(INFINI_ROOT .. "/lib", INFINI_ROOT .. "/lib64", { public = true })
    if is_plat("linux") then
        add_rpathdirs("$ORIGIN")
        -- A shared-library link otherwise tolerates missing provider symbols
        -- and only fails later at native executable link or Python import.
        -- Shared targets consume shflags; ldflags only affect binary links.
        add_shflags("-Wl,--no-undefined", {force = true})
    elseif is_plat("macosx") then
        add_rpathdirs("@loader_path")
    end
    add_links("infiniops", "infiniccl", "infinirt", { public = true })

    add_files("csrc/runtime/*.cc")
    add_files("csrc/runtime/tensor/*.cc")
    add_files("csrc/execution/trace/op_trace.cc")
    add_files("csrc/execution/graph/graph.cc")

    on_load(function (target)
        if not os.isfile(OPERATOR_CALLS_HEADER) then
            raise("InfiniOps operator instantiation manifest is missing: " .. OPERATOR_CALLS_HEADER)
        end
        local operator_calls = io.readfile(OPERATOR_CALLS_HEADER)
        for operator_type, requirement in pairs(DIRECT_OPS) do
            local symbol = "Operator<::infini::ops::" .. operator_type .. ">"
            if requirement and not operator_calls:find(symbol, 1, true) then
                if type(requirement) == "string" then
                    print("InfiniOps " .. operator_type .. " UNAVAILABLE for LM consumers"
                        .. " in profile " .. CAPABILITY_PROFILE .. "; retaining explicit rejection")
                    target:add("defines", requirement, { public = true })
                else
                    raise("InfiniOps " .. operator_type .. " is required by a public Ops LM call"
                        .. " in profile " .. CAPABILITY_PROFILE)
                end
            end
        end
    end)

    set_installdir("python/infinilm/_lib")
    on_install(function (target)
        os.mkdir(target:installdir())
        os.cp(target:targetfile(), target:installdir())
    end)
target_end()

target("_infinilm")
    add_linkdirs("/refactor/candidates/kimi-grouped-e2e-r1/lib")
    add_links("kimi_grouped_prefill")
    add_rpathdirs("/refactor/candidates/kimi-grouped-e2e-r1/lib")

    if not PYBIND11_INCLUDE_DIR then add_packages("pybind11") end
    set_default(false)
    add_rules("python.module", {soabi = true})
    set_languages("cxx17")
    set_kind("shared")

    add_deps("infinilm_runtime")

    if is_plat("linux") then
        add_rpathdirs("$ORIGIN/infinilm/_lib")
    elseif is_plat("macosx") then
        add_rpathdirs("@loader_path/infinilm/_lib")
    end

    add_includedirs("csrc")
    add_includedirs(INFINI_ROOT .. "/include")
    add_includedirs(INFINI_ROOT .. "/include/infiniccl")

    add_files("csrc/**.cpp|runtime/**.cpp|models/**.cpp")
    add_files("csrc/**.cc|runtime/**.cc|execution/graph/graph.cc|execution/trace/op_trace.cc|models/**.cc")
    add_files("csrc/models/*.cpp")
    add_files("csrc/models/baichuan/*.cpp")
    add_files("csrc/models/chatglm/*.cpp")
    add_files("csrc/models/deepseek/deepseek_for_causal_lm.cpp",
              "csrc/models/deepseek/deepseek_decoder_layer.cpp",
              "csrc/models/deepseek_v2/deepseek_v2_moe.cpp")
    add_files("csrc/models/ernie4_5_vl/*.cpp")
    add_files("csrc/models/fm9g/*.cpp")
    add_files("csrc/models/glm4/*.cpp")
    add_files("csrc/models/gpt2/gpt2_for_causal_lm.cpp")
    add_files("csrc/models/internlm3/*.cpp")
    add_files("csrc/models/llama/*.cpp")
    add_files("csrc/models/mamba/*.cpp")
    add_files("csrc/models/mistral/*.cpp")
    add_files("csrc/models/minicpm4/*.cpp")
    add_files("csrc/models/minicpm_eagle/*.cpp")
    add_files("csrc/models/qwen2/*.cpp")
    add_files("csrc/models/qwen3/*.cpp")
    add_files("csrc/models/kimi_k3/*.cpp")
    add_files("csrc/models/qwen3_moe/qwen3_moe_experts.cpp",
              "csrc/models/qwen3_moe/qwen3_moe_topk_router.cpp",
              "csrc/models/qwen3_moe/qwen3_moe_sparse_moe_block.cpp",
              "csrc/models/qwen3_moe/qwen3_moe_for_causal_lm.cpp")

    set_installdir("python")
    on_install(function (target)
        os.mkdir(target:installdir())
        os.cp(target:targetfile(), target:installdir())
    end)

target_end()

target("test_rank_barrier")
    set_default(false)
    set_kind("binary")
    set_languages("cxx17")
    add_includedirs("csrc/engine")
    add_files("test/native/rank_barrier_test.cpp")
    add_files("csrc/engine/rank_barrier.cpp")
    if is_plat("linux") then
        add_syslinks("pthread")
    end
target_end()

for _, name in ipairs({"tensor_empty_view"}) do
    target("test_" .. name)
        set_default(false)
        set_kind("binary")
        set_languages("cxx17")
        add_deps("infinilm_runtime")
        add_includedirs("csrc")
        add_files("test/native/" .. name .. "_test.cpp")
        if is_plat("linux") then
            add_syslinks("pthread")
        end
    target_end()
end

target("test_post_load_traversal")
    set_default(false)
    set_kind("binary")
    set_languages("cxx17")
    add_deps("infinilm_runtime")
    add_includedirs("csrc")
    add_files("test/native/post_load_traversal_test.cpp")
    add_files("csrc/weights/weights.cpp", "csrc/layers/embedding/embedding.cpp")
    add_files("csrc/layers/normalization/*.cpp")
    add_files("csrc/models/infinilm_model.cpp", "csrc/models/models_registry.cpp", "csrc/models/model_factory.cpp")
    add_files("csrc/models/minicpm4/minicpm4_for_causal_lm.cpp")
    add_files("csrc/cache/kv_cache.cpp")
    add_files("csrc/layers/linear/*.cpp", "csrc/layers/mlp/mlp.cpp")
    add_files("csrc/layers/attention/attention.cpp", "csrc/layers/attention/backends/*.cpp")
    add_files("csrc/layers/attention/kv_cache_quant_params.cpp")
    add_files("csrc/layers/rotary_embedding/*.cpp", "csrc/layers/quantization/kv_quant.cpp")
    add_files("csrc/config/model_config.cpp", "csrc/config/quant_config.cpp")
    add_files("csrc/layers/quantization/base_quantization.cpp", "csrc/layers/quantization/none_quantization.cpp")
    add_files("csrc/layers/quantization/awq.cpp", "csrc/layers/quantization/gptq.cpp")
    add_files("csrc/layers/quantization/w4_execution.cpp", "csrc/layers/quantization/w4_marlin_adapter.cpp")
    add_files("csrc/engine/distributed/communication_group.cpp",
              "csrc/engine/distributed/tcp_rendezvous.cpp",
              "csrc/engine/distributed/dist_config.cpp")
    if is_plat("linux") then
        add_syslinks("pthread")
    end
target_end()

target("test_parallel_topology")
    set_default(false)
    set_kind("binary")
    set_languages("cxx17")
    add_files("test/native/parallel_topology_test.cpp",
              "csrc/engine/distributed/dist_config.cpp")
target_end()
