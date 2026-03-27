#pragma once

#include "../backends/attention_backends.hpp"
#include "model_config.hpp"
#include <memory>

namespace infinilm::config {

/**
 * @brief  Dataclass which contains all infinilm-related configuration.
 *         This simplifies passing around the distinct configurations in the codebase.
 */
struct InfinilmConfig {
public:
    InfinilmConfig() = default;
    InfinilmConfig(const infinilm::backends::AttentionBackend &backend,
                   const std::shared_ptr<infinilm::config::ModelConfig> &model_config)
        : attention_backend(backend),
          model_config(model_config) {}

public:
    infinilm::backends::AttentionBackend attention_backend;
    std::shared_ptr<infinilm::config::ModelConfig> model_config;
};

/**
 * @brief  save the current Infinilm config in a global variable,
 *        so that all modules can access it, e.g. custom ops can access the Infinilm config to determine how to dispatch.
 */
void set_current_infinilm_config(const std::shared_ptr<InfinilmConfig> &config);

const InfinilmConfig &get_current_infinilm_config();

} // namespace infinilm::config
