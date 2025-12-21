#include "llama_model.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/ops.hpp"
#include <iostream>

namespace infinilm::models::llama {

LlamaModel::LlamaModel(const LlamaConfig &config,
                       const infinicore::Device &device,
                       infinicore::DataType dtype,
                       engine::distributed::RankInfo rank_info)
    : config_(config) {
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
            "layers." + std::to_string(i), config, device, i, dtype, rank_info));
    }

    // Initialize final layer normalization
    INFINICORE_NN_MODULE_INIT(norm, config.hidden_size, config.rms_norm_eps,
                              dtype, device);

    // Initialize Rotary Position Embeddings (shared across all layers)
    // Use GPT-J-style inverse frequencies (default) and GPT_NEOX rotation pairing
    INFINICORE_NN_MODULE_INIT(rotary_emb, config.head_dim, config.max_position_embeddings,
                              config.rope_theta, infinicore::nn::RoPE::Algo::GPT_NEOX,
                              dtype, device);

    for (auto &layer : layers_) {
        if (layer) {
            layer->set_rotary_emb(rotary_emb_);
        }
    }
}

infinicore::Tensor LlamaModel::forward(const infinicore::Tensor &input_ids,
                                       const infinicore::Tensor &position_ids,
                                       void *kv_cache) const {
    // Use persistent internal cache if no external cache is provided
    // This matches Python backend behavior: if use_cache and past_key_values is None, create DynamicCache
    // The cache persists across forward calls to enable incremental decoding
    void *cache_to_use = kv_cache;

    if (cache_to_use == nullptr) {
        // Create or reuse persistent internal cache at model level
        // This ensures the cache persists across multiple forward calls (prefill -> decode -> decode...)
        if (external_cache_ != nullptr) {
            cache_to_use = external_cache_;
        } else {
            // Fall back to internal cache
            if (!internal_cache_) {
                internal_cache_ = std::make_unique<infinilm::cache::DynamicCache>(
                    config_.num_hidden_layers,
                    config_.max_position_embeddings);
            }
            cache_to_use = internal_cache_.get();
        }
    }

    // 1. Embed tokens: input_ids -> [batch, seq_len, hidden_size]
    auto hidden_states = embed_tokens_->forward(input_ids);

    // 2. Process through all decoder layers
    size_t num_layers = layers_.size();
    for (size_t i = 0; i < num_layers; ++i) {
        // Pass model-level cache (layer index is now a property of the layer)
        hidden_states = layers_.at(i)->forward(hidden_states, position_ids, cache_to_use);

        // DEBUG: Disabled previous final layer logging
        // Logging moved to decoder layer for post-attention normalization
    }

    // 3. Apply final layer normalization to last token only (aligns with transformers)

    // Narrow to last token: [batch, seq_len, hidden_size] -> [batch, 1, hidden_size]
    auto shape = hidden_states->shape();
    size_t seq_len = shape[1];
    auto last_token = hidden_states->narrow({{1, seq_len - 1, 1}});

    // DEBUG: Disabled previous final layer normalization logging
    // Normalize only the last token (matches Python backend)
    auto normalized_last_token = norm_->forward(last_token);

    return normalized_last_token;
}

void LlamaModel::reset_cache(size_t pos) const {
    if (internal_cache_) {
        internal_cache_->reset(pos);
    }
    if (external_cache_) {
        external_cache_->reset(pos);
    }
}

void LlamaModel::reset_cache(const cache::CacheConfig &new_config, size_t pos) const {
    if (internal_cache_) {
        internal_cache_->update_config(new_config);
        internal_cache_->reset(pos);
    }
    if (external_cache_) {
        external_cache_->update_config(new_config);
        external_cache_->reset(pos);
    }
}

} // namespace infinilm::models::llama
