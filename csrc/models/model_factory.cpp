#include "model_factory.hpp"
#include "../global_state/forward_context.hpp"
#include "../layers/moe/ep/ep_config.hpp"
#include "models_registry.hpp"
#include "runtime/runtime.hpp"
#include <limits>

namespace infinilm {

void models::validate_runtime_construction(const std::shared_ptr<config::ModelConfig> &model_config,
                                           infinilm::runtime::Runtime &runtime,
                                           const engine::distributed::RankInfo &rank) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) {
        throw std::logic_error("model construction requires eager execution");
    }
    if (!model_config) {
        throw std::invalid_argument("model construction requires configuration");
    }
    if (rank.device != runtime.device() || rank.tp_size <= 0 || rank.tp_rank < 0 || rank.tp_rank >= rank.tp_size
        || rank.pp_size <= 0 || rank.pp_stage < 0 || rank.pp_stage >= rank.pp_size
        || rank.world_size <= 0 || rank.world_rank < 0 || rank.world_rank >= rank.world_size
        || static_cast<int64_t>(rank.world_size) != static_cast<int64_t>(rank.tp_size) * rank.pp_size
        || static_cast<int64_t>(rank.world_rank) != static_cast<int64_t>(rank.pp_stage) * rank.tp_size + rank.tp_rank
        || (rank.tp_size > 1 && !rank.comm) || (rank.pp_size > 1 && !rank.world_comm)) {
        throw std::invalid_argument("model construction rank topology is invalid");
    }
}

std::shared_ptr<InfinilmModel> InfinilmModelFactory::createModel(
    std::shared_ptr<config::ModelConfig> model_config, infinilm::runtime::Runtime &runtime,
    const engine::distributed::RankInfo &rank, backends::AttentionBackend backend,
    const layers::moe::EPConfig &ep, const cache::CacheConfig *cache,
    global_state::ForwardContext &fc, bool use_mla) {
    if (!model_config) {
        throw std::invalid_argument("Factory requires configuration");
    }
    runtime.assertOwnerThread();
    if (&fc.require_runtime() != &runtime) {
        throw std::invalid_argument("Factory cache context Runtime differs from construction Runtime");
    }
    const auto &map = models::get_causal_lm_model_map();
    const auto it = map.find(model_config->get<std::string>("model_type"));
    if (it == map.end()) {
        throw std::invalid_argument("ModelFactory unsupported model_type");
    }
    std::shared_ptr<InfinilmModel> model;
    try {
        model = it->second(model_config, runtime, rank, backend, ep, use_mla);
        if (cache) {
            model->reset_cache(cache, fc, backend, rank);
        }
        if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) {
            runtime.syncStream();
        }
        return model;
    } catch (...) {
        if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) {
            try {
                runtime.syncStream();
            } catch (...) { std::terminate(); }
        }
        throw;
    }
}
} // namespace infinilm
