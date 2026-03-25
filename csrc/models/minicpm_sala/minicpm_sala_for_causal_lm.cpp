#include "minicpm_sala_for_causal_lm.hpp"

#include "../../backends/attention_backends.hpp"
#include "../../cache/kv_cache.hpp"
#include "../../config/infinilm_config.hpp"
#include "../../engine/forward_context.hpp"
#include "../../engine/parallel_state.hpp"
#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::minicpm_sala {

MiniCPMSALAForCausalLM::MiniCPMSALAForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                               const infinicore::Device &device) {

    size_t hidden_size = model_config->get<size_t>("hidden_size");
    size_t vocab_size = model_config->get<size_t>("vocab_size");

    const auto &dtype{model_config->get_dtype()};

    INFINICORE_NN_MODULE_INIT(model, model_config, device);
    INFINICORE_NN_MODULE_INIT(lm_head, hidden_size, vocab_size, false, dtype, device);
}

infinilm::InfinilmModel::Output MiniCPMSALAForCausalLM::forward(const infinilm::InfinilmModel::Input &input) const {
    auto hidden_states = model_->forward(input);
    auto logits = lm_head_->forward(hidden_states);
    return {logits};
}

void MiniCPMSALAForCausalLM::reset_cache(const cache::CacheConfig *cache_config) {
    if (cache_config == nullptr) {
        InfinilmModel::reset_cache(nullptr);
        return;
    }
    auto &model_config = infinilm::config::get_current_infinilm_config().model_config;

    initialize_kv_cache_(cache_config, model_config);
}

void MiniCPMSALAForCausalLM::initialize_kv_cache_(const cache::CacheConfig *cache_config,
                                                  const std::shared_ptr<infinilm::config::ModelConfig> &text_config) {

    auto &forward_ctx = engine::get_forward_context();
    auto &kv_cache_vec = forward_ctx.kv_cache_vec;
    kv_cache_vec.clear();

    if (cache_config == nullptr) {
        return;
    }

    std::shared_ptr<infinilm::config::ModelConfig> effective_text_config = text_config;
    if (nullptr == effective_text_config) {
        effective_text_config = infinilm::config::get_current_infinilm_config().model_config;
    }
    if (nullptr == effective_text_config) {
        throw std::runtime_error("infinilm::InfinilmModel::initialize_kv_cache: text_config is null");
    }

    const backends::AttentionBackend attention_backend = infinilm::config::get_current_infinilm_config().attention_backend;
    const engine::distributed::RankInfo &rank_info = infinilm::engine::get_tensor_model_parallel_rank_info();

    switch (attention_backend) {
    case backends::AttentionBackend::STATIC_ATTN: {
        auto static_kv_cache_config = dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config);
        if (nullptr == static_kv_cache_config) {
            throw std::runtime_error("infinilm::InfinilmModel::initialize_kv_cache: invalid static kv cache config type");
        }
        const size_t num_hidden_layers = effective_text_config->get<size_t>("num_hidden_layers");
        kv_cache_vec.reserve(num_hidden_layers);

        const size_t head_dim = effective_text_config->get<size_t>("head_dim");
        const size_t num_key_value_heads = effective_text_config->get<size_t>("num_key_value_heads");
        const size_t max_position_embeddings = effective_text_config->get<size_t>("max_position_embeddings");

        const auto &dtype{effective_text_config->get_dtype()};
        std::vector<std::string> mixer_types = effective_text_config->get<std::vector<std::string>>("mixer_types");

        size_t current_layer_head_dim, current_layer_num_key_value_heads;
        for (size_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
            std::string mixer_type = mixer_types[layer_idx];

            if ("minicpm4" == mixer_type) {
                current_layer_head_dim = head_dim;
                current_layer_num_key_value_heads = num_key_value_heads;
            } else if ("lightning" == mixer_type || "lightning_attn" == mixer_type || "lightning-attn" == mixer_type) {
                current_layer_head_dim = effective_text_config->get<size_t>("lightning_head_dim");
                current_layer_num_key_value_heads = effective_text_config->get<size_t>("lightning_nkv");
            } else {
                throw std::runtime_error("infinilm::InfinilmModel::initialize_kv_cache: unsupported mixer_type '" + mixer_type + "' for layer " + std::to_string(layer_idx));
            }
            auto kv_cache = cache::StaticKVCache::create_layer_kv_cache(
                current_layer_head_dim,
                current_layer_head_dim,
                current_layer_num_key_value_heads,
                current_layer_num_key_value_heads,
                max_position_embeddings,
                dtype,
                *static_kv_cache_config,
                rank_info);

            kv_cache_vec.push_back(kv_cache);
        }
        break;
    }
    default:
        throw std::runtime_error("infinilm::InfinilmModel::initialize_kv_cache: Unsupported attention backend: " + std::to_string(static_cast<int>(attention_backend)));
    }
}

std::shared_ptr<infinilm::config::ModelConfig> create_minicpm_sala_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("minicpm_sala" != model_type) {
        throw std::runtime_error("infinilm::models::minicpm_sala::create_minicpm_sala_model_config: model_type is not minicpm_sala");
    }
    return model_config;
}

} // namespace infinilm::models::minicpm_sala
