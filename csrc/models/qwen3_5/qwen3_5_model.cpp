#include "qwen3_5_model.hpp"

#include "../../global_state/global_state.hpp"
#include "../qwen3_next/qwen3_next_allocate_kv_cache_tensors.hpp"

#include <utility>

namespace infinilm::models::qwen3_5 {

Qwen35Model::Qwen35Model(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         const infinicore::Device &device)
    : model_config_(model_config) {
    const auto &dtype{model_config->get_dtype()};
    nlohmann::json &config_json = model_config->get_config_json();

    if (config_json.contains("vision_config") && !config_json["vision_config"].is_null()) {
        INFINICORE_NN_MODULE_INIT(visual, config_json["vision_config"], dtype, device);
    }
    INFINICORE_NN_MODULE_INIT(language_model, model_config, device);
}

infinicore::Tensor Qwen35Model::forward(const InfinilmModel::Input &input) const {
    return language_model_->forward(input);
}

void Qwen35Model::reset_cache(const cache::CacheConfig *cache_config) {
    if (nullptr == cache_config) {
        return;
    }

    auto &forward_context = infinilm::global_state::get_forward_context();
    forward_context.kv_cache_vec.clear();
    forward_context.conv_state_vec.clear();
    forward_context.ssm_state_vec.clear();

    const backends::AttentionBackend attention_backend = infinilm::global_state::get_infinilm_config().attention_backend;

    auto cache_vectors = infinilm::models::qwen3_next::qwen3_next_allocate_cache_tensors(cache_config, model_config_, attention_backend);
    forward_context.kv_cache_vec = std::move(cache_vectors.kv_cache_tensors);
    forward_context.conv_state_vec = std::move(cache_vectors.conv_state_tensors);
    forward_context.ssm_state_vec = std::move(cache_vectors.ssm_state_tensors);
}

} // namespace infinilm::models::qwen3_5
