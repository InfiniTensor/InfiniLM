#pragma once

#include "../config/model_config.hpp"
#include "../engine/distributed/communication_group.hpp"
#include "../backends/attention_backends.hpp"
#include "../layers/moe/ep/ep_config.hpp"
#include "runtime/runtime.hpp"
#include <cmath>
#include <limits>
#include <cstdint>
#include "../weights/weights.hpp"
#include <type_traits>
#include <stdexcept>
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
 * @brief Factory that builds a causal LM instance from config and explicit Runtime resources.
 */
using ModelCreator = std::function<std::shared_ptr<InfinilmModel>(std::shared_ptr<config::ModelConfig>,
                                                                  infinilm::runtime::Runtime &, engine::distributed::RankInfo,
                                                                  backends::AttentionBackend, layers::moe::EPConfig, bool use_mla)>;

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
 * @brief Used by `INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL`: registers model factory + config handler at static init.
 *
 * @tparam ModelT Runtime-first model type, with exactly one complete constructor signature.
 * @tparam ConfigCreatorFn Type of a function like `create_qwen3_model_config` (for `decltype`).
 */
void validate_selected_construction(const std::shared_ptr<config::ModelConfig> &config,
                                   infinilm::runtime::Runtime &runtime,
                                   const engine::distributed::RankInfo &rank,
                                   const layers::moe::EPConfig &ep);

void validate_runtime_construction(const std::shared_ptr<config::ModelConfig> &config,
                                  infinilm::runtime::Runtime &runtime,
                                  const engine::distributed::RankInfo &rank,
                                  backends::AttentionBackend backend,
                                  const layers::moe::EPConfig &ep);

inline void validate_dense_construction_metadata(
    const std::shared_ptr<config::ModelConfig> &model_config, const char *family) {
    for (const char *key : {"hidden_size", "vocab_size", "num_hidden_layers", "head_dim", "num_attention_heads",
                           "num_key_value_heads", "intermediate_size", "max_position_embeddings"})
        if (model_config->get<int64_t>(key) <= 0) throw std::invalid_argument(std::string(family) + " requires positive dimensions");
    const auto dtype = model_config->get_dtype();
    if (dtype != infinilm::runtime::DataType::kFloat32 && dtype != infinilm::runtime::DataType::kFloat16 && dtype != infinilm::runtime::DataType::kBFloat16)
        throw std::invalid_argument(std::string(family) + " RoPE requires floating model dtype");
    const size_t limit = std::numeric_limits<size_t>::max(), bytes = infinilm::runtime::dsize(dtype);
    const auto product = [limit, family](size_t a, size_t b) {
        if (b && a > limit / b) throw std::length_error(std::string(family) + " construction size overflow");
        return a * b;
    };
    const size_t hidden = model_config->get<size_t>("hidden_size"), vocab = model_config->get<size_t>("vocab_size");
    const size_t heads = model_config->get<size_t>("num_attention_heads"), kv = model_config->get<size_t>("num_key_value_heads");
    const size_t dim = model_config->get<size_t>("head_dim"), mid = model_config->get<size_t>("intermediate_size");
    if (dim < 2 || heads % kv != 0) throw std::invalid_argument(std::string(family) + " GQA/RoPE geometry is invalid");
    const size_t q = product(heads, dim), k = product(kv, dim), twice_k = product(k, 2);
    if (q > limit - twice_k) throw std::length_error(std::string(family) + " packed QKV size overflow");
    (void)product(product(vocab, hidden), bytes);
    (void)product(product(q + twice_k, hidden), bytes); (void)product(product(q, hidden), bytes);
    (void)product(product(product(mid, 2), hidden), bytes); (void)product(product(mid, hidden), bytes);
    (void)product(hidden, bytes);
    (void)product(model_config->get<size_t>("num_hidden_layers"), sizeof(std::shared_ptr<infinilm::weights::ModelWeights>));
    const double epsilon = model_config->get<double>("rms_norm_eps"), theta = model_config->get<double>("rope_theta");
    const double factor = model_config->get_or<double>("partial_rotary_factor", 1.0);
    if (!std::isfinite(epsilon) || epsilon <= 0 || !std::isfinite(theta) || theta <= 0 || !std::isfinite(factor))
        throw std::invalid_argument(std::string(family) + " requires finite valid norm/RoPE metadata");
    if (factor > 0 && factor < 1 && static_cast<double>(dim) * factor >= static_cast<double>(std::numeric_limits<long long>::max()))
        throw std::length_error(std::string(family) + " rotary rounding overflow");
    const size_t rotary = model_config->get_rotary_dim();
    if (!rotary || rotary > dim || rotary % 2) throw std::invalid_argument(std::string(family) + " rotary geometry is invalid");
    const size_t cache = product(model_config->get<size_t>("max_position_embeddings"), rotary / 2);
    (void)product(product(cache, 2), sizeof(float)); (void)product(product(cache, 2), bytes);
}

template <typename ModelT, typename ConfigCreatorFn>
struct CausalLmRegistrar {
    /**
     * @brief Calls `register_causal_lm_model` and `register_model_config` for `model_type`.
     */
    explicit CausalLmRegistrar(const char *model_type, ConfigCreatorFn config_creator) {
        infinilm::models::register_causal_lm_model(
            model_type,
            [](std::shared_ptr<config::ModelConfig> config, infinilm::runtime::Runtime &runtime,
                             engine::distributed::RankInfo rank, backends::AttentionBackend backend,
                             layers::moe::EPConfig ep, bool use_mla) {
                constexpr bool without_mla = std::is_constructible_v<ModelT, infinilm::runtime::Runtime &,
                    std::shared_ptr<config::ModelConfig>, const engine::distributed::RankInfo &,
                    backends::AttentionBackend, const layers::moe::EPConfig &>;
                constexpr bool with_mla = std::is_constructible_v<ModelT, infinilm::runtime::Runtime &,
                    std::shared_ptr<config::ModelConfig>, const engine::distributed::RankInfo &,
                    backends::AttentionBackend, const layers::moe::EPConfig &, bool>;
                static_assert(without_mla != with_mla,
                    "model requires exactly one complete Runtime-first constructor, with or without explicit MLA");
                if constexpr (with_mla) {
                    return std::make_shared<ModelT>(runtime, std::move(config), rank, backend, ep, use_mla);
                } else {
                    return std::make_shared<ModelT>(runtime, std::move(config), rank, backend, ep);
                }
            });

        infinilm::models::register_model_config(model_type, config_creator);
    }
};

} // namespace infinilm::models

#define INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL(model_type, ModelT, ConfigCreatorFn) \
    auto g_##model_type##_registry = infinilm::models::CausalLmRegistrar<ModelT, decltype(ConfigCreatorFn)>(#model_type, ConfigCreatorFn)
