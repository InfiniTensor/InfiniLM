#pragma once

#include "../backends/attention_backends.hpp"
#include "../config/model_config.hpp"
#include <memory>

namespace infinilm::global_state {

/**
 * @brief Dataclass which contains all infinilm-related configuration.
 *        This simplifies passing around the distinct configurations in the codebase.
 */
struct InfinilmConfig {
public:
    InfinilmConfig() = default;
    InfinilmConfig(const infinilm::backends::AttentionBackend &backend,
                   const std::shared_ptr<infinilm::config::ModelConfig> &model_config)
        : attention_backend(backend),
          model_config(model_config) {

        const size_t max_position_embeddings = model_config->get<size_t>("max_position_embeddings");
        max_num_batched_tokens = max_position_embeddings;

        if (const char *max_num_batched_tokens_env = getenv("INFINILM_MAX_NUM_BATCHED_TOKENS")) {
            max_num_batched_tokens = std::stoi(max_num_batched_tokens_env);
            ASSERT(max_num_batched_tokens >= 1024 && max_num_batched_tokens <= max_position_embeddings);
        }
    }

public:
    infinilm::backends::AttentionBackend attention_backend;
    std::shared_ptr<infinilm::config::ModelConfig> model_config;
    size_t max_num_batched_tokens = -1;
};

/**
 * @brief save the current Infinilm config in a global variable,
 *        so that all modules can access it, e.g. custom ops can access the Infinilm config to determine how to dispatch.
 */
void initialize_infinilm_config(const std::shared_ptr<InfinilmConfig> &config);

const InfinilmConfig &get_infinilm_config();

} // namespace infinilm::global_state
