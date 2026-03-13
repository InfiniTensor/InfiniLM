#pragma once

#include "../backends/attention_backends.hpp"
#include "../cache/kv_cache.hpp"
#include "../config/model_config.hpp"

#include "../engine/distributed/distributed.hpp"
#include "infinicore/device.hpp"
#include "infinicore/io.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <optional>

namespace infinilm::models::layers {

/**
 * @brief Template decoder layer (transformer block) class
 *
 * A generic template class for decoder layers that can work with any attention and MLP types.
 * Each decoder layer consists of:
 * - Input layer normalization (RMSNorm)
 * - Self-attention mechanism
 * - Post-attention layer normalization (RMSNorm)
 * - MLP feed-forward network
 *
 * Residual connections are applied around both attention and MLP blocks.
 *
 * @tparam Attention The attention module type (e.g., LlamaAttention)
 * @tparam MLP The MLP module type (e.g., MLP)
 *
 * Usage example:
 * @code
 * using MyDecoderLayer = TemplateDecoderLayer<LlamaAttention, MLP>;
 * MyDecoderLayer layer(config, device, layer_idx, rank_info);
 * @endcode
 */
template <typename Attention, typename MLP>
class TemplateDecoderLayer : public infinicore::nn::Module {
public:
    /**
     * @brief Construct TemplateDecoderLayer module
     *
     * @param model_config Model configuration
     * @param device Device to create tensors on
     * @param layer_idx Layer index for cache management and debugging
     * @param rank_info Rank information for distributed training
     * @param attention_backend Attention backend to use
     */
    TemplateDecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         const infinicore::Device &device,
                         size_t layer_idx,
                         engine::distributed::RankInfo rank_info = engine::distributed::RankInfo(),
                         backends::AttentionBackend attention_backend = backends::AttentionBackend::Default)
        : model_config_(model_config), layer_idx_(layer_idx), rank_info_(rank_info) {
        const auto &dtype{model_config_->get_dtype()};
        // Initialize layer normalization layers
        input_layernorm_ = this->register_module<infinicore::nn::RMSNorm>("input_layernorm", model_config_->get<size_t>("hidden_size"), model_config_->get<double>("rms_norm_eps"),
                                                                          dtype, device);
        post_attention_layernorm_ = this->register_module<infinicore::nn::RMSNorm>("post_attention_layernorm", model_config_->get<size_t>("hidden_size"), model_config_->get<double>("rms_norm_eps"),
                                                                                   dtype, device);

        // Initialize attention and MLP modules
        self_attn_ = this->register_module<Attention>("self_attn", model_config_, device, layer_idx, rank_info_, attention_backend);
        mlp_ = this->register_module<MLP>("mlp", model_config_, device, rank_info_);
    }

    /**
     * @brief Forward pass: process one decoder layer
     *
     * @param hidden_states [batch, seq_len, hidden_size], will be modified
     * @param residual [batch, seq_len, hidden_size], will be modified
     * @param position_ids Position IDs tensor of shape [batch, seq_len] or [seq_len]
     * @param kv_cache Optional KV cache for incremental decoding
     * @param past_sequence_lengths Cache positions tensor of shape [n_req]
     * @param total_sequence_lengths Total sequence lengths tensor of shape [n_req]
     * @param input_offsets Input offsets (starting position) of each request in a continuous batch of shape [n_req + 1]
     * @param cu_seqlens Cumulative total sequence lengths for each request, of shape [n_req + 1]
     * @param block_tables Block ids for each request [batch, max_block_table_length]. Used for paged cache.
     * @param slot_mapping Slot ids for each token [seq]. Used for paged cache.
     * @return Output tensor of shape [batch, seq_len, hidden_size]
     *         Updated residual tensor of shape [batch, seq_len, hidden_size]
     */
    std::tuple<infinicore::Tensor, infinicore::Tensor>
    forward(infinicore::Tensor &hidden_states,
            infinicore::Tensor &residual,
            const infinicore::Tensor &position_ids,
            std::shared_ptr<infinilm::cache::Cache> kv_cache,
            std::optional<infinicore::Tensor> past_sequence_lengths,
            std::optional<infinicore::Tensor> total_sequence_lengths,
            std::optional<infinicore::Tensor> input_offsets,
            std::optional<infinicore::Tensor> cu_seqlens,
            std::optional<infinicore::Tensor> block_tables,
            std::optional<infinicore::Tensor> slot_mapping) const {

        // 1. Attention layer normalization
        input_layernorm_->forward_inplace(hidden_states, residual);

        // 2. Self-attention
        hidden_states = self_attn_->forward(
            hidden_states, position_ids, kv_cache, past_sequence_lengths, total_sequence_lengths, input_offsets, cu_seqlens, block_tables, slot_mapping);

        // 3. Post-attention layer normalization
        post_attention_layernorm_->forward_inplace(hidden_states, residual);

        // 4. MLP
        hidden_states = mlp_->forward(hidden_states);
        if (0) {
            std::cout << "--------------> mlp_ hidden_states ::: " << std::endl;
            infinicore::print_options::set_sci_mode(1);
            std::cout << hidden_states << std::endl;
        }
        return std::make_tuple(hidden_states, residual);
    }

    /**
     * @brief Get the layer index
     */
    size_t layer_idx() const { return layer_idx_; }

    void set_rotary_emb(const std::shared_ptr<infinicore::nn::RoPE> &rotary_emb) {
        if (self_attn_) {
            self_attn_->set_rotary_emb(rotary_emb);
        }
    }

protected:
    // Layer normalization
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);

    // Attention and MLP
    INFINICORE_NN_MODULE(Attention, self_attn);
    INFINICORE_NN_MODULE(MLP, mlp);
    engine::distributed::RankInfo rank_info_;
    std::shared_ptr<infinilm::config::ModelConfig> model_config_;

private:
    size_t layer_idx_; // Layer index for cache management and debugging
};

} // namespace infinilm::models::layers
