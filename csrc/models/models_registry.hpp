#pragma once

#include "../backends/attention_backends.hpp"
#include "../config/model_config.hpp"
#include "../engine/distributed/distributed.hpp"
#include "infinicore/device.hpp"
#include "infinilm_model.hpp"
#include <functional>
#include <map>
#include <memory>
#include <string>

namespace infinilm::models {

using ModelCreator = std::function<std::shared_ptr<InfinilmModel>(
    std::shared_ptr<config::ModelConfig>,
    const infinicore::Device &,
    engine::distributed::RankInfo,
    backends::AttentionBackend)>;

using ConfigCreator = std::function<std::shared_ptr<config::ModelConfig>(
    std::shared_ptr<config::ModelConfig>)>;

void register_causal_lm_models(std::map<std::string, ModelCreator> &map);

void register_model_configs(std::map<std::string, ConfigCreator> &map);

} // namespace infinilm::models
