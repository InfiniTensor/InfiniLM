#pragma once

#include "../config/model_config.hpp"
#include "infinilm_model.hpp"

#include "../backends/attention_backends.hpp"
#include "../engine/distributed/distributed.hpp"

#include "infinicore/device.hpp"

#include <functional>
#include <map>

namespace infinilm {

/** @brief Factory function type: creates a model from config, device, rank_info, and attention backend */
using ModelCreator = std::function<std::shared_ptr<InfinilmModel>(
    std::shared_ptr<config::ModelConfig>,
    const infinicore::Device &,
    engine::distributed::RankInfo,
    backends::AttentionBackend)>;

class InfinilmModelFactory {
public:
    static std::shared_ptr<InfinilmModel> createModel(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        engine::distributed::RankInfo rank_info = engine::distributed::RankInfo(),
        const cache::CacheConfig *cache = nullptr,
        backends::AttentionBackend attention_backend = backends::AttentionBackend::Default);

private:
    static std::map<std::string, ModelCreator> &_modelsForCausalLM();
};

} // namespace infinilm
