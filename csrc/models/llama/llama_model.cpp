#include "llama_model.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/ops.hpp"
#include <iostream>

namespace infinilm::models::llama {

LlamaModel::LlamaModel(const infinicore::Device &device,
                       engine::distributed::RankInfo rank_info,
                       std::shared_ptr<infinilm::config::global_config::GlobalConfig> global_config)
    : rank_info_(rank_info), global_config_(global_config) {
    const auto &dtype{global_config_->get_dtype()};
    // Initialize token embeddings
    INFINICORE_NN_MODULE_INIT(embed_tokens, global_config_->get<size_t>("vocab_size"), global_config_->get<size_t>("hidden_size"),
                              std::nullopt, dtype, device);
    // Initialize decoder layers with layer indices
    // TODO: Update INFINICORE_NN_MODULE_VEC_INIT macro to support per-layer constructor arguments
    //       (e.g., via a factory function or lambda that receives the layer index)
    //       Currently, we can't use the macro because each layer needs a different layer_idx
    layers_.reserve(global_config_->get<size_t>("num_hidden_layers"));
    for (size_t i = 0; i < global_config_->get<size_t>("num_hidden_layers"); ++i) {
        layers_.push_back(this->register_module<LlamaDecoderLayer>(
            "layers." + std::to_string(i), device, i, rank_info, global_config_));
    }

    // Initialize final layer normalization
    INFINICORE_NN_MODULE_INIT(norm, global_config_->get<size_t>("hidden_size"), global_config_->get<double>("rms_norm_eps"),
                              dtype, device);

    // Initialize Rotary Position Embeddings (shared across all layers)
    // Use GPT-J-style inverse frequencies (default) and GPT_NEOX rotation pairing
    INFINICORE_NN_MODULE_INIT(rotary_emb, global_config_->get_head_dim(), global_config_->get<size_t>("max_position_embeddings"),
                              global_config_->get<double>("rope_theta"), infinicore::nn::RoPE::Algo::GPT_NEOX,
                              dtype, device, global_config_->get_rope_scaling());

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
    if (auto kv_cache_config = dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config)) {
        kv_cache_ = std::make_shared<cache::StaticKVCache>(
            global_config_->get_head_dim(),
            global_config_->get_head_dim(),
            global_config_->get<size_t>("num_key_value_heads"),
            global_config_->get<size_t>("num_key_value_heads"),
            global_config_->get<size_t>("num_hidden_layers"),
            global_config_->get<size_t>("max_position_embeddings"),
            global_config_->get_dtype(),
            *kv_cache_config,
            rank_info_);
    } else if (auto paged_kv_cache_config = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config)) {
        kv_cache_ = std::make_shared<cache::PagedKVCache>(
            global_config_->get_head_dim(),
            global_config_->get_head_dim(),
            global_config_->get<size_t>("num_key_value_heads"),
            global_config_->get<size_t>("num_key_value_heads"),
            global_config_->get<size_t>("num_hidden_layers"),
            global_config_->get_dtype(),
            *paged_kv_cache_config,
            rank_info_);
    } else {
        throw std::runtime_error("Unsupported cache type");
    }
}

} // namespace infinilm::models::llama
