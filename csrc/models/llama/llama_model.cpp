#include "llama_model.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/ops.hpp"
#include <iostream>

namespace infinilm::models::llama {
/**
 * @deprecated This function is deprecated and will be REMOVED in the next major release (v0.2.0).
 *
 * ⚠️ DEVELOPMENT POLICY:
 *   - NO new development or feature additions permitted on this interface
 *   - Only critical bug fixes (security/stability) allowed until removal
 *   - All new code MUST migrate to the polymorphic overload below
 *
 * Replacement: Use the polymorphic overload of this same function name with updated signature
 * Reason: Legacy signature lacks support for dynamic quantization modes.
 * Removal target: v0.2.0 (Q2 2026)
 */
LlamaModel::LlamaModel(const LlamaConfig &config,
                       const infinicore::Device &device,
                       engine::distributed::RankInfo rank_info)
    : config_(config), rank_info_(rank_info) {
    const auto &dtype{config.dtype};
    // Initialize token embeddings
    INFINICORE_NN_MODULE_INIT(embed_tokens, config.vocab_size, config.hidden_size,
                              std::nullopt, dtype, device);

    // Initialize decoder layers with layer indices
    // TODO: Update INFINICORE_NN_MODULE_VEC_INIT macro to support per-layer constructor arguments
    //       (e.g., via a factory function or lambda that receives the layer index)
    //       Currently, we can't use the macro because each layer needs a different layer_idx
    layers_.reserve(config.num_hidden_layers);
    for (size_t i = 0; i < config.num_hidden_layers; ++i) {
        layers_.push_back(this->register_module<LlamaDecoderLayer>(
            "layers." + std::to_string(i), config, device, i, rank_info));
    }

    // Initialize final layer normalization
    INFINICORE_NN_MODULE_INIT(norm, config.hidden_size, config.rms_norm_eps,
                              dtype, device);

    // Initialize Rotary Position Embeddings (shared across all layers)
    // Use GPT-J-style inverse frequencies (default) and GPT_NEOX rotation pairing
    INFINICORE_NN_MODULE_INIT(rotary_emb, config.head_dim, config.max_position_embeddings,
                              config.rope_theta, infinicore::nn::RoPE::Algo::GPT_NEOX,
                              dtype, device, config.rope_scaling);

    for (auto &layer : layers_) {
        if (layer) {
            layer->set_rotary_emb(rotary_emb_);
        }
    }
}

LlamaModel::LlamaModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                       const infinicore::Device &device,
                       engine::distributed::RankInfo rank_info)
    : model_config_(model_config), rank_info_(rank_info) {
    const auto &dtype{model_config_->get_dtype()};
    // Initialize token embeddings
    INFINICORE_NN_MODULE_INIT(embed_tokens, model_config_->get<size_t>("vocab_size"), model_config_->get<size_t>("hidden_size"),
                              std::nullopt, dtype, device);
    // Initialize decoder layers with layer indices
    // TODO: Update INFINICORE_NN_MODULE_VEC_INIT macro to support per-layer constructor arguments
    //       (e.g., via a factory function or lambda that receives the layer index)
    //       Currently, we can't use the macro because each layer needs a different layer_idx
    layers_.reserve(model_config_->get<size_t>("num_hidden_layers"));
    for (size_t i = 0; i < model_config_->get<size_t>("num_hidden_layers"); ++i) {
        layers_.push_back(this->register_module<LlamaDecoderLayer>(
            "layers." + std::to_string(i), model_config_, device, i, rank_info));
    }
    // Initialize final layer normalization
    INFINICORE_NN_MODULE_INIT(norm, model_config_->get<size_t>("hidden_size"), model_config_->get<double>("rms_norm_eps"),
                              dtype, device);
    // Initialize Rotary Position Embeddings (shared across all layers)
    // Use GPT-J-style inverse frequencies (default) and GPT_NEOX rotation pairing
    INFINICORE_NN_MODULE_INIT(rotary_emb, model_config_->get_head_dim(), model_config_->get<size_t>("max_position_embeddings"),
                              model_config_->get<double>("rope_theta"), infinicore::nn::RoPE::Algo::GPT_NEOX,
                              dtype, device, model_config_->get_rope_scaling());

    for (auto &layer : layers_) {
        if (layer) {
            layer->set_rotary_emb(rotary_emb_);
        }
    }
}

infinicore::Tensor LlamaModel::forward(const infinicore::Tensor &input_ids,
                                       const infinicore::Tensor &position_ids,
                                       std::optional<infinicore::Tensor> past_sequence_lengths,
                                       std::optional<infinicore::Tensor> total_sequence_lengths,
                                       std::optional<infinicore::Tensor> input_offsets,
                                       std::optional<infinicore::Tensor> block_tables,
                                       std::optional<infinicore::Tensor> slot_mapping) const {
    // 1. Embed tokens: input_ids -> [batch, seq_len, hidden_size]
    auto hidden_states = embed_tokens_->forward(input_ids);

    // 2. Process through all decoder layers
    size_t num_layers = layers_.size();
    infinicore::Tensor residual;
    for (size_t i = 0; i < num_layers; ++i) {
        layers_.at(i)->forward(
            hidden_states,
            residual,
            position_ids,
            kv_cache_,
            past_sequence_lengths,
            total_sequence_lengths,
            input_offsets,
            block_tables,
            slot_mapping);
    }

    norm_->forward_inplace(hidden_states, residual);

    return hidden_states;
}

void LlamaModel::reset_cache(const cache::CacheConfig *cache_config) {
    if (cache_config == nullptr) {
        kv_cache_ = nullptr;
        return;
    }
    if (auto kv_cache_config = dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config);
        kv_cache_config && model_config_ == nullptr) {
        kv_cache_ = std::make_shared<cache::StaticKVCache>(
            config_.head_dim,
            config_.head_dim,
            config_.num_key_value_heads,
            config_.num_key_value_heads,
            config_.num_hidden_layers,
            config_.max_position_embeddings,
            config_.dtype,
            *kv_cache_config,
            rank_info_);
    } else if (auto paged_kv_cache_config = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config);
               paged_kv_cache_config && model_config_ == nullptr) {
        kv_cache_ = std::make_shared<cache::PagedKVCache>(
            config_.head_dim,
            config_.head_dim,
            config_.num_key_value_heads,
            config_.num_key_value_heads,
            config_.num_hidden_layers,
            config_.dtype,
            *paged_kv_cache_config,
            rank_info_);
    } else if (auto kv_cache_config = dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config)) {
        kv_cache_ = std::make_shared<cache::StaticKVCache>(
            model_config_->get_head_dim(),
            model_config_->get_head_dim(),
            model_config_->get<size_t>("num_key_value_heads"),
            model_config_->get<size_t>("num_key_value_heads"),
            model_config_->get<size_t>("num_hidden_layers"),
            model_config_->get<size_t>("max_position_embeddings"),
            model_config_->get_dtype(),
            *kv_cache_config,
            rank_info_);
    } else if (auto paged_kv_cache_config = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config)) {
        kv_cache_ = std::make_shared<cache::PagedKVCache>(
            model_config_->get_head_dim(),
            model_config_->get_head_dim(),
            model_config_->get<size_t>("num_key_value_heads"),
            model_config_->get<size_t>("num_key_value_heads"),
            model_config_->get<size_t>("num_hidden_layers"),
            model_config_->get_dtype(),
            *paged_kv_cache_config,
            rank_info_);
    } else {
        throw std::runtime_error("Unsupported cache type");
    }
}

} // namespace infinilm::models::llama
