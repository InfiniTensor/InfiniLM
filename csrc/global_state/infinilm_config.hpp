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
                   const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
                   bool use_mla = false)
        : attention_backend(backend),
          use_mla(use_mla),
          model_config(model_config) {

        if (max_num_batched_tokens > 0) {
            const size_t max_position_embeddings = model_config->get<size_t>("max_position_embeddings");
            ASSERT(max_num_batched_tokens >= 512 && max_num_batched_tokens <= max_position_embeddings);
            enable_workspace_manager = true;
        }
        enable_workspace_manager = false;
    }

public:
    infinilm::backends::AttentionBackend attention_backend;
    bool use_mla{false};
    std::shared_ptr<infinilm::config::ModelConfig> model_config;
    size_t max_num_batched_tokens = 0;
    bool enable_workspace_manager{false};
};

/**
 * @brief save the current Infinilm config in a global variable,
 *        so that all modules can access it, e.g. custom ops can access the Infinilm config to determine how to dispatch.
 */
void initialize_infinilm_config(const std::shared_ptr<InfinilmConfig> &config);

const InfinilmConfig &get_infinilm_config();

} // namespace infinilm::global_state
