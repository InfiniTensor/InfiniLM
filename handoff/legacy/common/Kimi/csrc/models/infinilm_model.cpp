#include "../weights/weights.hpp"
#include "../layers/embedding/embedding.hpp"
#include "../layers/normalization/rmsnorm.hpp"
#include "infinilm_model.hpp"
#include "../cache/kv_cache.hpp"
#include "../global_state/global_state.hpp"
#include "../utils.hpp"
#include "runtime/runtime.hpp"
#include <exception>
#include <stdexcept>
#include <typeinfo>
#include <limits>
#include "qwen3/qwen3_for_causal_lm.hpp"
#include "../parallel/topology.hpp"

namespace infinilm {

void InfinilmModel::reset_cache(const cache::CacheConfig *cache_config,
                               global_state::ForwardContext &forward_context,
                               backends::AttentionBackend attention_backend,
                               const engine::distributed::RankInfo &rank) {
    reset_default_cache(cache_config, forward_context, attention_backend, rank);
}

void InfinilmModel::reset_default_cache(
    const cache::CacheConfig *cache_config,
    global_state::ForwardContext &forward_context,
    backends::AttentionBackend attention_backend,
    const engine::distributed::RankInfo &rank) {
    cache::validate_backend_cache_config(cache_config, attention_backend);
    if (cache_config == nullptr) {
        cache_config_.reset();
        forward_context.kv_cache_vec.clear();
        return;
    }
    auto &runtime = forward_context.require_runtime();
    cache::validate_cache_runtime(runtime, rank);
    cache_config_ = cache_config->unique_copy();
    auto &kv_cache_vec = forward_context.kv_cache_vec;
    kv_cache_vec.clear();
    kv_cache_vec = std::move(default_allocate_kv_cache_tensors(runtime, rank, cache_config, model_config_, attention_backend));
}

std::vector<infinilm::runtime::Tensor> InfinilmModel::default_allocate_kv_cache_tensors(
    infinilm::runtime::Runtime &runtime,
    const engine::distributed::RankInfo &rank,
    const cache::CacheConfig *cache_config,
    const std::shared_ptr<infinilm::config::ModelConfig> &text_config,
    const backends::AttentionBackend &attention_backend) {
    if (nullptr == cache_config) {
        return {};
    }
    cache::validate_backend_cache_config(cache_config, attention_backend);
    cache::validate_cache_runtime(runtime, rank);
    if (nullptr == text_config) {
        throw std::runtime_error("infinilm::InfinilmModel::default_allocate_kv_cache_tensors: text_config is null");
    }
    size_t head_dim = text_config->get<size_t>("head_dim");
    size_t num_key_value_heads = text_config->get<size_t>("num_key_value_heads");
    size_t max_position_embeddings = text_config->get<size_t>("max_position_embeddings");
    const auto &dtype = model_config_->get_kv_cache_dtype();
    const size_t num_hidden_layers = text_config->get<size_t>("num_hidden_layers");
    const parallel::Topology topology(static_cast<size_t>(rank.tp_size), rank.pp_size, rank.pp_stage);
    const auto [local_layer_begin, local_layer_end] = topology.layer_range(num_hidden_layers);

    std::vector<infinilm::runtime::Tensor> kv_cache_vec;
    try {
    switch (attention_backend) {
    case backends::AttentionBackend::STATIC_ATTN: {
        auto static_kv_cache_config = dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config);
        if (nullptr == static_kv_cache_config) {
            throw std::runtime_error("infinilm::InfinilmModel::default_allocate_kv_cache_tensors: invalid static kv cache config type");
        }
        kv_cache_vec.resize(num_hidden_layers);

        for (size_t layer_idx = local_layer_begin; layer_idx < local_layer_end; ++layer_idx) {
            auto kv_cache = cache::StaticKVCache::create_layer_kv_cache(
                    runtime,
                    rank,
                head_dim,
                head_dim,
                num_key_value_heads,
                num_key_value_heads,
                max_position_embeddings,
                dtype,
                *static_kv_cache_config);
            kv_cache_vec[layer_idx] = kv_cache;
        }
        break;
    }
    case backends::AttentionBackend::FLASH_ATTN: {
        ;
    }
    case backends::AttentionBackend::PAGED_ATTN: {
        auto paged_kv_cache_config = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config);
        if (nullptr == paged_kv_cache_config) {
            throw std::runtime_error(
                "infinilm::InfinilmModel::default_allocate_kv_cache_tensors: invalid paged kv cache config type");
        }
        kv_cache_vec.resize(num_hidden_layers);

        for (size_t layer_idx = local_layer_begin; layer_idx < local_layer_end; ++layer_idx) {
            auto kv_cache = cache::PagedKVCache::create_layer_kv_cache(
                    runtime,
                    rank,
                head_dim,
                head_dim,
                num_key_value_heads,
                num_key_value_heads,
                dtype,
                *paged_kv_cache_config);
            kv_cache_vec[layer_idx] = kv_cache;
        }
            runtime.syncStream();
        break;
    }
    default:
        throw std::runtime_error("infinilm::InfinilmModel::default_allocate_kv_cache_tensors: Unsupported attention backend: " + std::to_string(static_cast<int>(attention_backend)));
    }
    } catch (...) {
        // Previously completed layer destinations remain held by the local vector.
        try {
            runtime.syncStream();
        } catch (...) {
            std::terminate();
        }
        throw;
    }
    return kv_cache_vec;
}

namespace {
using Module = infinilm::weights::ModelWeights;
void w4_require(bool ok, const char *message) { if (!ok) throw std::invalid_argument(message); }
void w4_keys(const Module &owner, const std::vector<std::string> &keys) {
    w4_require(owner.children().size() == keys.size(), "AWQ exact owner children count mismatch");
    for (const auto &key : keys) w4_require(owner.children().count(key) == 1, "AWQ missing owner child");
}
template<class T> std::shared_ptr<T> w4_child(const Module &owner, const std::string &name) {
    const auto it = owner.children().find(name);
    w4_require(it != owner.children().end() && it->second && typeid(*it->second) == typeid(T),
                "AWQ unsupported concrete owner child");
    return std::static_pointer_cast<T>(it->second);
    }
}

std::unique_ptr<quantization::W4ModelPlan> InfinilmModel::validate_w4_weights(
    infinilm::runtime::Runtime &runtime, bool pre_transpose,
    const engine::distributed::RankInfo &rank,
    const std::string &ep_backend, size_t ep_size) {
    runtime.assertOwnerThread();
    w4_require(runtime.graphCaptureState() == infinilm::execution::graph::GraphManager::CaptureState::kInactive,
                "W4 validation requires inactive graph state");
    w4_require(runtime.device().type() == infinilm::runtime::Device::Type::kNvidia, "AWQ requires NVIDIA Runtime");
    w4_require(!pre_transpose,
                "W4 finalization does not support the legacy dense pre_transpose path");
    using namespace models::qwen3;
    w4_require(model_config_ != nullptr, "W4 model config is required");
    const auto expected_scheme = model_config_->get_quant_scheme();
    w4_require(typeid(*this) == typeid(Qwen3ForCausalLM) &&
                model_config_->get<std::string>("model_type") == "qwen3" &&
                (expected_scheme == quantization::QuantScheme::AWQ_W4A16 ||
                 expected_scheme == quantization::QuantScheme::GPTQ_W4A16),
                "W4 unsupported model owner family");
    w4_require(runtime.device() == rank.device, "AWQ Runtime does not match current worker rank device");
    if (expected_scheme == quantization::QuantScheme::GPTQ_W4A16) {
        w4_require((rank.tp_size == 1 || rank.tp_size == 2) && rank.pp_size == 1 &&
                   ep_backend == "disabled" && ep_size == 1,
                   "GPTQ owner requires TP1/2, PP1 and disabled EP");
    }
    const auto range = parallel::Topology(rank.tp_size, rank.pp_size, rank.pp_stage)
                           .layer_range(model_config_->get<size_t>("num_hidden_layers"));
    const bool first = rank.pp_stage == 0, last = rank.pp_stage + 1 == rank.pp_size;
    w4_require(range.second - range.first <= std::numeric_limits<size_t>::max() / 4,
                "AWQ local coverage count overflow");
    w4_keys(*this, last ? std::vector<std::string>{"model", "lm_head"} : std::vector<std::string>{"model"});
    auto text = w4_child<Qwen3Model>(*this, "model");
    std::vector<std::string> model_keys;
    if (first) model_keys.push_back("embed_tokens");
    if (last) model_keys.push_back("norm");
    for (size_t i = range.first; i < range.second; ++i) model_keys.push_back("layers." + std::to_string(i));
    w4_keys(*text, model_keys);
    if (first) (void)w4_child<infinilm::layers::Embedding>(*text, "embed_tokens");
    if (last) (void)w4_child<infinilm::layers::RMSNorm>(*text, "norm");
    std::vector<std::shared_ptr<nn::BaseLinear>> layers;
    layers.reserve((range.second - range.first) * 4);
    for (size_t i = range.first; i < range.second; ++i) {
        auto decoder = w4_child<Qwen3DecoderLayer>(*text, "layers." + std::to_string(i));
        w4_keys(*decoder, {"input_layernorm", "post_attention_layernorm", "self_attn", "mlp"});
        (void)w4_child<infinilm::layers::RMSNorm>(*decoder, "input_layernorm");
        (void)w4_child<infinilm::layers::RMSNorm>(*decoder, "post_attention_layernorm");
        auto attention = w4_child<Qwen3Attention>(*decoder, "self_attn");
        w4_keys(*attention, {"q_norm", "k_norm", "o_proj"});
        (void)w4_child<infinilm::layers::RMSNorm>(*attention, "q_norm");
        (void)w4_child<infinilm::layers::RMSNorm>(*attention, "k_norm");
        attention->append_w4_layers(layers, expected_scheme); // exact hidden QKV strong owner
        layers.push_back(w4_child<nn::RowParallelLinear>(*attention, "o_proj"));
        auto mlp = w4_child<Qwen3MLP>(*decoder, "mlp");
        w4_keys(*mlp, {"down_proj"});
        mlp->append_w4_layers(layers, expected_scheme); // exact hidden GateUp strong owner
        layers.push_back(w4_child<nn::RowParallelLinear>(*mlp, "down_proj"));
    }
    w4_require(layers.size() == (range.second - range.first) * 4, "AWQ missing layer coverage");
    std::shared_ptr<nn::BaseLinear> dense_head;
    if (last) {
        auto head = w4_child<nn::Linear>(*this, "lm_head");
        w4_require(head->get_quantization()->get_quant_scheme() == quantization::QuantScheme::NONE,
                    "AWQ lm_head must retain the existing dense owner");
        dense_head = std::move(head);
    }
    // Constructor validates EVERY layer and duplicate before returning a plan;
    // no device allocation/repack, no raw callbacks and no noarg fused hook here.
    auto plan = std::make_unique<quantization::W4ModelPlan>(expected_scheme, std::move(layers), std::move(dense_head));
    plan->validate_metadata(runtime); // all local entries before this rank's VALIDATE acknowledgement
    return plan;
}

void InfinilmModel::process_weights_after_loading(infinilm::runtime::Runtime &runtime, bool pre_transpose) {
    runtime.assertOwnerThread();
    for (const auto &[_, sub] : children()) {
        process_weights_recursive_(sub.get(), runtime, pre_transpose);
    }
}

void InfinilmModel::process_weights_recursive_(infinilm::weights::ModelWeights *module,
                                               infinilm::runtime::Runtime &runtime, bool pre_transpose) {
    // Nested causal LMs own their traversal; do not visit their children twice.
    if (auto *nested = dynamic_cast<InfinilmModel *>(module)) {
        nested->process_weights_after_loading(runtime, pre_transpose);
        return;
    }
    for (const auto &[_, sub] : module->children()) {
        process_weights_recursive_(sub.get(), runtime, pre_transpose);
    }
    module->process_weights_after_loading(runtime, pre_transpose);
}

} // namespace infinilm
