#include "infinilm_model.hpp"
#include "../cache/kv_cache.hpp"
#include "../global_state/global_state.hpp"
#include "../layers/embedding/embedding.hpp"
#include "../layers/normalization/rmsnorm.hpp"
#include "../utils.hpp"
#include "../weights/weights.hpp"
#include "runtime/runtime.hpp"
#include <exception>
#include <limits>
#include <stdexcept>
#include <unordered_set>

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
    if (!forward_context.kv_cache_vec.empty()) {
        auto &runtime = forward_context.require_runtime();
        cache::validate_cache_runtime(runtime, rank);
        // Capacity changes are an idle boundary, not the per-token hot path.
        // Drain before releasing backing, then reclaim unpinned idle blocks
        // before allocating a differently sized cache.
        runtime.syncStream();
        forward_context.kv_cache_vec.clear();
        runtime.trimMemory();
    }
    if (cache_config == nullptr) {
        cache_config_.reset();
        return;
    }
    auto &runtime = forward_context.require_runtime();
    cache::validate_cache_runtime(runtime, rank);
    cache_config_ = cache_config->unique_copy();
    auto &kv_cache_vec = forward_context.kv_cache_vec;
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
void w4_require(bool ok, const char *message) {
    if (!ok) {
        throw std::invalid_argument(message);
    }
}
void collect_w4(const Module &owner, quantization::QuantScheme scheme,
                std::vector<std::shared_ptr<nn::BaseLinear>> &layers,
                std::vector<std::shared_ptr<nn::BaseLinear>> &dense,
                std::vector<std::shared_ptr<Module>> &hooks,
                std::unordered_set<const Module *> &visited) {
    for (const auto &child : owner.loading_children()) {
        w4_require(bool(child), "Null loading module");
        if (!visited.insert(child.get()).second) {
            continue;
        }
        if (auto linear = std::dynamic_pointer_cast<nn::BaseLinear>(child)) {
            if (linear->get_quantization()->get_quant_scheme() == quantization::QuantScheme::NONE) {
                dense.push_back(std::move(linear));
            } else {
                w4_require(linear->get_quantization()->get_quant_scheme() == scheme, "Mixed W4 layer scheme");
                layers.push_back(std::move(linear));
            }
        } else {
            collect_w4(*child, scheme, layers, dense, hooks, visited);
            // Nested models orchestrate traversal; they have no owner-local hook.
            if (!std::dynamic_pointer_cast<InfinilmModel>(child)) {
                hooks.push_back(child);
            }
        }
    }
}
} // namespace

std::unique_ptr<quantization::W4ModelPlan> InfinilmModel::validate_w4_weights(
    infinilm::runtime::Runtime &runtime, bool pre_transpose,
    const engine::distributed::RankInfo &rank,
    const std::string &ep_backend, size_t ep_size) {
    runtime.assertOwnerThread();
    w4_require(runtime.graphCaptureState() == infinilm::execution::graph::GraphManager::CaptureState::kInactive,
               "W4 validation requires inactive graph state");
    w4_require(!pre_transpose,
               "W4 finalization does not support the legacy dense pre_transpose path");
    w4_require(model_config_ != nullptr, "W4 model config is required");
    const auto expected_scheme = model_config_->get_quant_scheme();
    w4_require(runtime.device() == rank.device, "W4 Runtime does not match worker device");
    std::vector<std::shared_ptr<nn::BaseLinear>> layers, dense;
    std::vector<std::shared_ptr<Module>> hooks;
    std::unordered_set<const Module *> visited{this};
    collect_w4(*this, expected_scheme, layers, dense, hooks, visited);
    // Constructor validates EVERY layer and duplicate before returning a plan;
    // no device allocation/repack, no raw callbacks and no noarg fused hook here.
    auto plan = std::make_unique<quantization::W4ModelPlan>(expected_scheme, std::move(layers), std::move(dense), std::move(hooks));
    plan->validate_metadata(runtime); // all local entries before this rank's VALIDATE acknowledgement
    return plan;
}

void InfinilmModel::process_weights_after_loading(infinilm::runtime::Runtime &runtime, bool pre_transpose) {
    runtime.assertOwnerThread();
    std::unordered_set<const weights::ModelWeights *> visited{this};
    for (const auto &sub : loading_children()) {
        process_weights_recursive_(sub.get(), runtime, pre_transpose, visited);
    }
}

void InfinilmModel::process_weights_recursive_(infinilm::weights::ModelWeights *module,
                                               infinilm::runtime::Runtime &runtime, bool pre_transpose,
                                               std::unordered_set<const weights::ModelWeights *> &visited) {
    if (!module || !visited.insert(module).second) {
        return;
    }
    for (const auto &sub : module->loading_children()) {
        process_weights_recursive_(sub.get(), runtime, pre_transpose, visited);
    }
    if (!dynamic_cast<InfinilmModel *>(module)) {
        module->process_weights_after_loading(runtime, pre_transpose);
    }
}

} // namespace infinilm
