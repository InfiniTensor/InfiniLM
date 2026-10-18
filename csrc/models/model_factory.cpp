#include "model_factory.hpp"
#include "runtime/runtime.hpp"
#include "models_registry.hpp"
#include "../global_state/forward_context.hpp"
#include "../layers/moe/ep/ep_config.hpp"
#include <limits>

namespace infinilm {

void models::validate_selected_construction(const std::shared_ptr<config::ModelConfig> &model_config,
                                           infinilm::runtime::Runtime &runtime,
                                           const engine::distributed::RankInfo &rank,
                                           const layers::moe::EPConfig &ep) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("selected construction requires eager Runtime");
    if (!model_config) throw std::invalid_argument("selected constructor requires configuration");
    const auto model_type = model_config->get<std::string>("model_type");
    const bool is_qwen3_text = model_type == "qwen3" || model_type == "qwen3_vl_text";
    if (model_type != "qwen3_moe" && !is_qwen3_text && model_type != "llama" && model_type != "qwen2" && model_type != "mistral") throw std::invalid_argument("selected constructor requires an explicitly migrated family");
    if ((model_type == "qwen3_moe" || is_qwen3_text) && model_config->get_kv_quant_scheme() != quantization::KVQuantAlgo::NONE)
        throw std::invalid_argument("selected Qwen constructor requires NONE KV quantization");
    if ((model_type == "llama" || model_type == "qwen2" || model_type == "mistral") && model_config->get_kv_quant_scheme() != quantization::KVQuantAlgo::NONE
        && model_config->get_kv_quant_scheme() != quantization::KVQuantAlgo::INT8)
        throw std::invalid_argument("unsupported Llama kv_quant_scheme");
    if (ep.backend != layers::moe::EPBackend::Disabled) throw std::invalid_argument("selected constructor requires disabled EP");
        if (rank.device != runtime.device() || rank.tp_size <= 0 || rank.tp_rank < 0 || rank.tp_rank >= rank.tp_size
            || rank.pp_size <= 0 || rank.pp_stage < 0 || rank.pp_stage >= rank.pp_size
            || rank.world_size <= 0 || rank.world_rank < 0 || rank.world_rank >= rank.world_size
            || static_cast<int64_t>(rank.world_size) != static_cast<int64_t>(rank.tp_size) * rank.pp_size
            || static_cast<int64_t>(rank.world_rank) != static_cast<int64_t>(rank.pp_stage) * rank.tp_size + rank.tp_rank
            || (rank.tp_size > 1 && !rank.comm) || (rank.pp_size > 1 && !rank.world_comm)) throw std::invalid_argument("Qwen3MoE Factory rank topology is invalid");
        const size_t tp = static_cast<size_t>(rank.tp_size);
        const size_t heads = model_config->get<size_t>("num_attention_heads");
        const size_t kv_heads = model_config->get<size_t>("num_key_value_heads");
        if (is_qwen3_text || model_type == "llama" || model_type == "qwen2" || model_type == "mistral") {
            if (model_config->get<size_t>("intermediate_size") % tp != 0 || heads % tp != 0
                || (kv_heads >= tp ? kv_heads % tp != 0 : tp % kv_heads != 0)
                || (rank.pp_size > 1 && model_config->get<size_t>("hidden_size") % tp != 0))
                throw std::invalid_argument("Qwen3 Factory TP geometry is invalid");
            // QKV replicates KV heads when TP exceeds KV count: check its exact packed expression before Embedding.
            const size_t limit = std::numeric_limits<size_t>::max();
            const auto product = [limit](size_t a, size_t b) {
                if (b && a > limit / b) throw std::length_error("Qwen3 replicated QKV size overflow");
                return a * b;
            };
            const size_t dim = model_config->get<size_t>("head_dim");
            const size_t replication = kv_heads < tp ? tp / kv_heads : 1;
            const size_t q = product(heads, dim), k = product(product(kv_heads, dim), replication);
            const size_t twice_k = product(k, 2);
            if (q > limit - twice_k) throw std::length_error("Qwen3 replicated packed QKV size overflow");
            (void)product(product(q + twice_k, model_config->get<size_t>("hidden_size")), infinilm::runtime::dsize(model_config->get_dtype()));
        } else if (model_config->get<size_t>("moe_intermediate_size") % tp != 0 || model_config->get<size_t>("hidden_size") % tp != 0
            || heads % tp != 0 || (kv_heads >= tp ? kv_heads % tp != 0 : tp % kv_heads != 0)) throw std::invalid_argument("Qwen3MoE Factory TP geometry is invalid");
    }

void models::validate_runtime_construction(const std::shared_ptr<config::ModelConfig> &model_config,
                                          infinilm::runtime::Runtime &runtime,
                                          const engine::distributed::RankInfo &rank,
                                          backends::AttentionBackend backend,
                                          const layers::moe::EPConfig &ep) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("Runtime construction requires eager execution");
    if (!model_config) throw std::invalid_argument("Runtime constructor requires configuration");
    const auto type = model_config->get<std::string>("model_type");
    if (type == "qwen3_moe" || type == "qwen3" || type == "llama" || type == "qwen2" || type == "mistral") {
        validate_selected_construction(model_config, runtime, rank, ep);
        return;
    }
    const bool hybrid = type == "qwen3_next" || type == "qwen3_5" || type == "qwen3_5_moe";
    const bool remaining = type == "gpt2" || type == "deepseek" || type == "deepseek_v2";
    const bool vision = type == "qwen3_vl" || type == "qwen3_vl_text" || type == "minicpmv"
        || type == "ernie4_5_moe_vl" || type == "videonsa" || type == "kimi_k25" || type == "kimi_k3";
    if (!hybrid && !remaining && !vision && type != "baichuan" && type != "chatglm" && type != "fm9g" && type != "fm9g7b"
        && type != "minicpm" && type != "glm4" && type != "internlm3" && type != "minicpm4"
        && type != "minicpm_eagle" && type != "mamba")
        throw std::invalid_argument("Runtime constructor requires an explicitly migrated family");
    if (!hybrid && !vision && ep.backend != layers::moe::EPBackend::Disabled)
        throw std::invalid_argument("Runtime constructor requires disabled EP");
    if (rank.device != runtime.device() || rank.tp_size <= 0 || rank.tp_rank < 0 || rank.tp_rank >= rank.tp_size
        || rank.pp_size <= 0 || rank.pp_stage < 0 || rank.pp_stage >= rank.pp_size
        || rank.world_size <= 0 || rank.world_rank < 0 || rank.world_rank >= rank.world_size
        || static_cast<int64_t>(rank.world_size) != static_cast<int64_t>(rank.tp_size) * rank.pp_size
        || static_cast<int64_t>(rank.world_rank) != static_cast<int64_t>(rank.pp_stage) * rank.tp_size + rank.tp_rank
        || (rank.tp_size > 1 && !rank.comm) || (rank.pp_size > 1 && !rank.world_comm))
        throw std::invalid_argument("Runtime constructor rank topology is invalid");
    if (hybrid || remaining || vision) {
        return;
    }
    const size_t limit = std::numeric_limits<size_t>::max();
    const auto product = [limit](size_t a, size_t b) {
        if (b && a > limit / b) throw std::length_error("Runtime construction size overflow");
        return a * b;
    };
    const auto positive = [&model_config](const char *key) {
        if (model_config->get<int64_t>(key) <= 0)
            throw std::invalid_argument(std::string("Runtime constructor requires positive ") + key);
        return model_config->get<size_t>(key);
    };
    if (type != "mamba") {
        // Existing family handlers divide these raw operands to default head_dim.
        (void)positive("hidden_size");
        (void)positive("num_attention_heads");
        if (backend != backends::AttentionBackend::STATIC_ATTN
            && backend != backends::AttentionBackend::PAGED_ATTN
            && backend != backends::AttentionBackend::FLASH_ATTN)
            throw std::invalid_argument("Unsupported explicit Attention backend");
    }
    if (type == "mamba") return; // Its own handler owns the complete Mamba metadata contract.
    validate_dense_construction_metadata(model_config, type.c_str());
    const size_t tp = static_cast<size_t>(rank.tp_size);
    const size_t heads = model_config->get<size_t>("num_attention_heads"), kv = model_config->get<size_t>("num_key_value_heads");
    if (model_config->get<size_t>("intermediate_size") % tp != 0 || heads % tp != 0
        || (kv >= tp ? kv % tp != 0 : tp % kv != 0))
        throw std::invalid_argument("Runtime constructor TP geometry is invalid");
    if (type != "minicpm4" && type != "minicpm_eagle" && rank.pp_size > 1
        && model_config->get<size_t>("hidden_size") % tp != 0)
        throw std::invalid_argument("TextModel pipeline hidden geometry is invalid");
    const size_t dim = model_config->get<size_t>("head_dim");
    const size_t q = product(heads, dim), k = product(product(kv, dim), kv < tp ? tp / kv : 1), twice_k = product(k, 2);
    if (q > limit - twice_k) throw std::length_error("Runtime replicated packed QKV size overflow");
    (void)product(product(q + twice_k, model_config->get<size_t>("hidden_size")), infinilm::runtime::dsize(model_config->get_dtype()));
    (void)model_config->get_quantization_method();
    const auto kv_scheme = model_config->get_kv_quant_scheme();
    if (kv_scheme != quantization::KVQuantAlgo::NONE && kv_scheme != quantization::KVQuantAlgo::INT8)
        throw std::invalid_argument("unsupported kv_quant_scheme");
    for (const char *key : {"scale_depth", "dim_model_base"}) {
        if (model_config->get_config_json().contains(key) && !std::isfinite(model_config->get<double>(key)))
            throw std::invalid_argument(std::string("Runtime constructor requires finite ") + key);
    }
    if (type == "minicpm_eagle") {
        const double denominator = model_config->get_or<double>("mup_denominator", 1.0);
        if (!std::isfinite(denominator) || denominator <= 0)
            throw std::invalid_argument("Eagle requires finite positive MuP denominator");
        (void)product(product(product(model_config->get<size_t>("hidden_size"), 2),
                              model_config->get<size_t>("hidden_size")), infinilm::runtime::dsize(model_config->get_dtype()));
    }

}

std::shared_ptr<InfinilmModel> InfinilmModelFactory::createModel(
    std::shared_ptr<config::ModelConfig> model_config, infinilm::runtime::Runtime &runtime,
    const engine::distributed::RankInfo &rank, backends::AttentionBackend backend,
    const layers::moe::EPConfig &ep, const cache::CacheConfig *cache,
    global_state::ForwardContext &fc, bool use_mla) {
    if (!model_config) throw std::invalid_argument("Factory requires configuration");
    runtime.assertOwnerThread();
    if (&fc.require_runtime() != &runtime) throw std::invalid_argument("Factory cache context Runtime differs from construction Runtime");
    const auto &map = models::get_causal_lm_model_map();
    const auto it = map.find(model_config->get<std::string>("model_type"));
    if (it == map.end()) throw std::invalid_argument("ModelFactory unsupported model_type");
    std::shared_ptr<InfinilmModel> model;
    try {
        model = it->second(model_config, runtime, rank, backend, ep, use_mla);
        if (cache) model->reset_cache(cache, fc, backend, rank);
        if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) runtime.syncStream();
    return model;
    } catch (...) {
        if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) {
            try { runtime.syncStream(); } catch (...) { std::terminate(); }
        }
        throw;
}
}
} // namespace infinilm
