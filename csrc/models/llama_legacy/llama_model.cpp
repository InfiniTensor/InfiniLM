#include "llama_model.hpp"
#include "../../layers/rotary_embedding/rotary_embedding_factory.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/ops.hpp"
#include <iostream>

namespace infinilm::models::llama_legacy {

LlamaModel::LlamaModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                       const infinicore::Device &device,
                       engine::distributed::RankInfo rank_info,
                       backends::AttentionBackend attention_backend)
    : model_config_(model_config), rank_info_(rank_info) {
    const auto &dtype{model_config_->get_dtype()};
    INFINICORE_NN_MODULE_INIT(embed_tokens, model_config_->get<size_t>("vocab_size"), model_config_->get<size_t>("hidden_size"),
                              std::nullopt, dtype, device);
    layers_.reserve(model_config_->get<size_t>("num_hidden_layers"));
    for (size_t i = 0; i < model_config_->get<size_t>("num_hidden_layers"); ++i) {
        layers_.push_back(this->register_module<LlamaDecoderLayer>(
            "layers." + std::to_string(i), model_config_, device, i, rank_info, attention_backend));
    }
    INFINICORE_NN_MODULE_INIT(norm, model_config_->get<size_t>("hidden_size"), model_config_->get<double>("rms_norm_eps"),
                              dtype, device);
    auto rope_scaling_config = infinilm::layers::rotary_embedding::make_scaling_config(model_config_);
    INFINICORE_NN_MODULE_INIT(rotary_emb, model_config_->get_head_dim(), model_config->get_rotary_dim(), model_config_->get<size_t>("max_position_embeddings"),
                              model_config_->get<double>("rope_theta"), infinicore::nn::RoPE::Algo::GPT_NEOX,
                              dtype, device, rope_scaling_config);

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
                                       std::optional<infinicore::Tensor> cu_seqlens,
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
            cu_seqlens,
            block_tables,
            slot_mapping);
    }

    norm_->forward_inplace(hidden_states, residual);

    return hidden_states;
}

infinicore::Tensor LlamaModel::forward_embeds(const infinicore::Tensor &inputs_embeds,
                                              const infinicore::Tensor &position_ids,
                                              std::optional<infinicore::Tensor> past_sequence_lengths,
                                              std::optional<infinicore::Tensor> total_sequence_lengths,
                                              std::optional<infinicore::Tensor> input_offsets,
                                              std::optional<infinicore::Tensor> cu_seqlens,
                                              std::optional<infinicore::Tensor> block_tables,
                                              std::optional<infinicore::Tensor> slot_mapping) const {
    auto hidden_states = inputs_embeds;
    size_t num_layers = layers_.size();
    infinicore::Tensor residual;
    for (size_t i = 0; i < num_layers; ++i) {
        layers_.at(i)->forward(hidden_states, residual, position_ids, kv_cache_, past_sequence_lengths, total_sequence_lengths, input_offsets, cu_seqlens, block_tables, slot_mapping);
    }
    norm_->forward_inplace(hidden_states, residual);

    return hidden_states;
}

infinicore::Tensor LlamaModel::embed_tokens(const infinicore::Tensor &input_ids) const {
    return embed_tokens_->forward(input_ids);
}

void LlamaModel::reset_cache(const cache::CacheConfig *cache_config) {
    if (cache_config == nullptr) {
        kv_cache_ = nullptr;
        return;
    }
    if (auto kv_cache_config = dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config)) {
        kv_cache_ = std::make_shared<cache::StaticKVCache>(
            model_config_->get_head_dim(),
            model_config_->get_head_dim(),
            model_config_->get<size_t>("num_key_value_heads"),
            model_config_->get<size_t>("num_key_value_heads"),
            model_config_->get<size_t>("num_hidden_layers"),
            model_config_->get<size_t>("max_position_embeddings"),
            model_config_->get_kv_cache_dtype(),
            *kv_cache_config,
            rank_info_);
    } else if (auto paged_kv_cache_config = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config)) {
        kv_cache_ = std::make_shared<cache::PagedKVCache>(
            model_config_->get_head_dim(),
            model_config_->get_head_dim(),
            model_config_->get<size_t>("num_key_value_heads"),
            model_config_->get<size_t>("num_key_value_heads"),
            model_config_->get<size_t>("num_hidden_layers"),
            model_config_->get_kv_cache_dtype(),
            *paged_kv_cache_config,
            rank_info_);
    } else {
        throw std::runtime_error("Unsupported cache type");
    }
}

} // namespace infinilm::models::llama_legacy
