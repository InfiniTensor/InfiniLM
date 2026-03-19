#pragma once

#include "../../backends/attention_backends.hpp"
#include "../../cache/kv_cache.hpp"
#include "../../config/model_config.hpp"
#include "../../engine/distributed/distributed.hpp"
#include "qwen3_next_gated_deltanet.hpp"

#include "../infinilm_model.hpp"
#include "infinicore/device.hpp"
#include "infinicore/io.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/tensor.hpp"
#include "qwen3_next_attention.hpp"
#include "qwen3_next_sparse_moe_block.hpp"
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>

namespace infinilm::models::qwen3_next {

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
 * @tparam Attention The attention module type (variant of InfLLMv2Attention/LightningAttention)
 * @tparam MLP The MLP module type (e.g., MiniCPMMLP)
 *
 * Usage example:
 * @code
 * using MyDecoderLayer = Qwen3NextDecoderLayer<MiniCPMSALAAttention, MiniCPMMLP>;
 * MyDecoderLayer layer(config, device, layer_idx, rank_info);
 * @endcode
 */

class Qwen3NextDecoderLayer : public infinicore::nn::Module {
public:
    /**
     * @brief Construct Qwen3NextDecoderLayer module
     *
     * @param model_config Model configuration
     * @param device Device to create tensors on
     * @param layer_idx Layer index for cache management and debugging
     * @param rank_info Rank information for distributed training
     * @param attention_backend Reserved (unused; attention type selected via config mixer_types)
     */
    Qwen3NextDecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                          size_t layer_idx,
                          const infinicore::Device &device,
                          engine::distributed::RankInfo rank_info = engine::distributed::RankInfo(),
                          backends::AttentionBackend attention_backend = backends::AttentionBackend::Default)
        : model_config_(model_config), layer_idx_(layer_idx), rank_info_(rank_info) {
        const auto &dtype{model_config_->get_dtype()};
        size_t hidden_size = model_config_->get<size_t>("hidden_size");
        size_t intermediate_size = model_config_->get<size_t>("intermediate_size");
        double rms_norm_eps = model_config_->get<double>("rms_norm_eps");

        // Initialize layer normalization layers
        input_layernorm_ = this->register_module<infinicore::nn::RMSNorm>("input_layernorm", hidden_size, rms_norm_eps, dtype, device);
        post_attention_layernorm_ = this->register_module<infinicore::nn::RMSNorm>("post_attention_layernorm", hidden_size, rms_norm_eps, dtype, device);

        // Initialize MLP
        mlp_ = this->register_module<Qwen3NextSparseMoeBlock>("mlp", model_config_, device, rank_info_);

        // Initialize attention
        std::vector<std::string> layer_types = model_config_->get<std::vector<std::string>>("layer_types");
        layer_type_ = layer_types[layer_idx];
        if ("linear_attention" == layer_type_) {
            linear_attn_ = this->register_module<Qwen3NextGatedDeltaNet>("linear_attn", model_config_, layer_idx, device, rank_info_);
        } else if ("full_attention" == layer_type_) {
            self_attn_ = this->register_module<Qwen3NextAttention>("self_attn", model_config_, layer_idx, device, rank_info_, attention_backend);
        }
    }

    /**
     * @brief Forward pass: process one decoder layer
     *
     * @param hidden_states [batch, seq_len, hidden_size], will be modified
     * @param residual [batch, seq_len, hidden_size], will be modified
     * @param input Encapsulated input tensors and other parameters
     * @param kv_cache Optional KV cache for incremental decoding
     * @return Output tensor of shape [batch, seq_len, hidden_size]
     *         Updated residual tensor of shape [batch, seq_len, hidden_size]
     */
    std::tuple<infinicore::Tensor, infinicore::Tensor>
    forward(infinicore::Tensor &hidden_states,
            infinicore::Tensor &residual,
            const infinilm::InfinilmModel::Input &input,
            std::shared_ptr<infinilm::cache::Cache> kv_cache) {

        // 1. Attention layer normalization
        input_layernorm_->forward_inplace(hidden_states, residual);

        // 2. attention
        if ("linear_attention" == layer_type_) {
            hidden_states = linear_attn_->forward(hidden_states);
        } else if ("full_attention" == layer_type_) {
            hidden_states = self_attn_->forward(hidden_states, input, kv_cache);
        }

        // 3. Post-attention layer normalization
        post_attention_layernorm_->forward_inplace(hidden_states, residual);

        // 4. MLP
        hidden_states = mlp_->forward(hidden_states);
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
    INFINICORE_NN_MODULE(Qwen3NextAttention, self_attn);
    INFINICORE_NN_MODULE(Qwen3NextGatedDeltaNet, linear_attn);
    INFINICORE_NN_MODULE(Qwen3NextSparseMoeBlock, mlp);
    engine::distributed::RankInfo rank_info_;
    std::shared_ptr<infinilm::config::ModelConfig> model_config_;

private:
    size_t layer_idx_;
    std::string layer_type_;
};

} // namespace infinilm::models::qwen3_next
