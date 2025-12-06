#include "llama_model.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/ops.hpp"

namespace infinilm::models::llama {

LlamaModel::LlamaModel(const LlamaConfig &config, const infinicore::Device &device,
                       infinicore::DataType dtype)
    : config_(config) {
    // Initialize token embeddings
    INFINICORE_NN_MODULE_INIT(embed_tokens, config.vocab_size, config.hidden_size,
                              std::nullopt, dtype, device);

    // Initialize decoder layers
    INFINICORE_NN_MODULE_VEC_INIT(layers, config.num_hidden_layers, LlamaDecoderLayer,
                                   config, device, dtype);

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

    if (kv_cache == nullptr) {
        // Create or reuse persistent internal cache at model level
        // This ensures the cache persists across multiple forward calls (prefill -> decode -> decode...)
        size_t seq_len = input_ids->shape()[1];

        if (!cache_) {
            // First time: create cache
            cache_ = std::make_unique<infinilm::cache::DynamicCache>(
                config_.num_hidden_layers,
                config_.max_position_embeddings
            );
        }
        cache_to_use = cache_.get();
    }

    // 1. Embed tokens: input_ids -> [batch, seq_len, hidden_size]
    auto hidden_states = embed_tokens_->forward(input_ids);

    // 2. Process through all decoder layers
    size_t num_layers = layers_.size();
    for (size_t i = 0; i < num_layers; ++i) {
        // Pass model-level cache with layer index
        hidden_states = layers_.at(i)->forward(hidden_states, position_ids, cache_to_use, i);

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

void LlamaModel::reset_cache(bool full_reset) const {
    if (cache_) {
        if (full_reset) {
            cache_->full_reset();
        } else {
            cache_->reset();
        }
    }
}

} // namespace infinilm::models::llama
