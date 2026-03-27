#pragma once

#include "infinicore/device.hpp"
#include <functional>
#include <map>
#include <memory>
#include <string>

namespace infinilm::config {
class ModelConfig;
}

namespace infinilm {
class InfinilmModel;
}

namespace infinilm::models {

/**
 * @brief Factory that builds a causal LM instance from config and device.
 */
using ModelCreator = std::function<std::shared_ptr<InfinilmModel>(std::shared_ptr<config::ModelConfig>,
                                                                  const infinicore::Device &)>;

/**
 * @brief  post-processor for `ModelConfig`.
 */
using ConfigCreator = std::function<std::shared_ptr<config::ModelConfig>(std::shared_ptr<config::ModelConfig>)>;

/**
 * @brief Register one causal LM constructor for a `model_type` string.
 */
void register_causal_lm_model(const std::string &model_type, ModelCreator creator);

/**
 * @brief Register one config post-processor for a `model_type` string.
 */
void register_model_config(const std::string &model_type, ConfigCreator creator);

/**
 * @brief Snapshot of all registered causal LM factories.
 *
 * @return Map from `model_type` to `ModelCreator`.
 */
const std::map<std::string, ModelCreator> &get_causal_lm_model_map();

/**
 * @brief Snapshot of all registered config post-processors.
 *
 * @return Map from `model_type` to `ConfigCreator`.
 */
const std::map<std::string, ConfigCreator> &get_model_config_map();

/**
 * @brief Used by `INFINILM_REGISTER_CAUSAL_LM_MODEL`: registers model factory + config handler at static init.
 *
 * @tparam ModelT Causal LM type constructible as `std::make_shared<ModelT>(config, device)`.
 * @tparam ConfigCreatorFn Type of a function like `create_qwen3_model_config` (for `decltype`).
 */
template <typename ModelT, typename ConfigCreatorFn>
struct CausalLmRegistrar {
    /**
     * @brief Calls `register_causal_lm_model` and `register_model_config` for `model_type`.
     */
    explicit CausalLmRegistrar(const char *model_type, ConfigCreatorFn config_creator) {
        infinilm::models::register_causal_lm_model(
            model_type,
            [](std::shared_ptr<config::ModelConfig> config, const infinicore::Device &device) {
                return std::make_shared<ModelT>(std::move(config), device);
            });

        infinilm::models::register_model_config(model_type, config_creator);
    }
};

} // namespace infinilm::models

#define INFINILM_REGISTER_CAUSAL_LM_MODEL(model_type, ModelT, ConfigCreatorFn) \
    auto g_##model_type##_registry = infinilm::models::CausalLmRegistrar<ModelT, decltype(ConfigCreatorFn)>(#model_type, ConfigCreatorFn)
