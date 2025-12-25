#include "llama_model.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/ops.hpp"
#include <iostream>

namespace infinilm::models::llama {

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
                              dtype, device);

    for (auto &layer : layers_) {
        if (layer) {
            layer->set_rotary_emb(rotary_emb_);
        }
    }
}

infinicore::Tensor LlamaModel::forward(const infinicore::Tensor &input_ids,
                                       const infinicore::Tensor &position_ids,
                                       const infinicore::Tensor &cache_positions) const {
    // 1. Embed tokens: input_ids -> [batch, seq_len, hidden_size]
    auto hidden_states = embed_tokens_->forward(input_ids);

    // 2. Process through all decoder layers
    // Reuse residual across layers to avoid redundant add operations
    size_t num_layers = layers_.size();
    std::optional<infinicore::Tensor> residual = std::nullopt;
    for (size_t i = 0; i < num_layers; ++i) {
        auto [output, next_residual] = layers_.at(i)->forward(hidden_states, position_ids, kv_cache_, cache_positions, residual);
        hidden_states = output;
        residual = next_residual;
    }

    // 3. Apply final layer normalization to last token only (aligns with transformers)
    // Narrow to last token: [batch, seq_len, hidden_size] -> [batch, 1, hidden_size]
    auto shape = hidden_states->shape();
    size_t seq_len = shape[1];
    
    // Narrow both residual and mlp_output to last token before fusing add and norm
    // Note: narrow() creates a view (no data copy), so this is equivalent to:
    //   narrow(add(residual, mlp_output)) == add(narrow(residual), narrow(mlp_output))
    // But doing narrow first allows us to:
    //   1. Only compute add for the last token (not the entire sequence) - saves computation
    //   2. Fuse add with norm in a single kernel using add_rms_norm - avoids separate add kernel
    auto residual_last_token = residual.value()->narrow({{1, seq_len - 1, 1}});
    auto mlp_output_last_token = hidden_states->narrow({{1, seq_len - 1, 1}});
    
    // Fuse final residual add with layer normalization using add_rms_norm
    // This avoids a separate add operation - add and norm are computed in one fused kernel
    // Result is mathematically equivalent to: norm(add(residual, mlp_output))[last_token]
    auto [normalized_last_token, _] = infinicore::op::add_rms_norm(
        residual_last_token, mlp_output_last_token,
        norm_->weight(),
        static_cast<float>(norm_->eps()));

    return normalized_last_token;
}

void LlamaModel::reset_cache(const cache::CacheConfig *cache_config) {
    if (cache_config == nullptr) {
        kv_cache_ = nullptr;
        return;
    }
    if (auto kv_cache_config = dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config)) {
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

    } else {
        throw std::runtime_error("Unsupported cache type");
    }
}

} // namespace infinilm::models::llama
