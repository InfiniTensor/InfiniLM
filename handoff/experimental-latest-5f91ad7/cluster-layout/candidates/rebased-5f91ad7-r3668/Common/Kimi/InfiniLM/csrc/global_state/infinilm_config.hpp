#pragma once

#include "../backends/attention_backends.hpp"
#include "../config/model_config.hpp"
#include <cstddef>
#include <memory>
#include <string>
#include <utility>

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
                   bool use_mla = false,
                   std::string moe_ep_backend = "disabled",
                   size_t moe_ep_size = 1,
                   bool pre_transpose = false)
        : attention_backend(backend),
          use_mla(use_mla),
          moe_ep_backend(std::move(moe_ep_backend)),
          moe_ep_size(moe_ep_size),
          pre_transpose(pre_transpose),
          model_config(model_config) {}

public:
    infinilm::backends::AttentionBackend attention_backend;
    bool use_mla{false};
    std::string moe_ep_backend{"disabled"};
    size_t moe_ep_size{1};
    bool pre_transpose{false};
    std::shared_ptr<infinilm::config::ModelConfig> model_config;
};

} // namespace infinilm::global_state
