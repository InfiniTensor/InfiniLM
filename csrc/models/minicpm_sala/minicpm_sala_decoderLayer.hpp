#pragma once

#include "../../backends/attention_backends.hpp"
#include "../../config/model_config.hpp"
#include "../../engine/distributed/distributed.hpp"

#include "../infinilm_model.hpp"
#include "infinicore/device.hpp"
#include "infinicore/io.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/tensor.hpp"
#include "minicpm_sala_attention.hpp"
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>

namespace infinilm::models::minicpm_sala {

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
 * using MyDecoderLayer = MiniCPMSALADecoderLayer<MiniCPMSALAAttention, MiniCPMMLP>;
 * MyDecoderLayer layer(config, device, layer_idx, rank_info);
 * @endcode
 */
template <typename Attention, typename MLP>
class MiniCPMSALADecoderLayer : public infinicore::nn::Module {
public:
    /**
     * @brief Construct MiniCPMSALADecoderLayer module
     *
     * @param model_config Model configuration
     * @param device Device to create tensors on
     * @param layer_idx Layer index for cache management and debugging
     * @param rank_info Rank information for distributed training
     */
    MiniCPMSALADecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                            size_t layer_idx,
                            const infinicore::Device &device)
        : model_config_(model_config), layer_idx_(layer_idx) {
        const auto &dtype{model_config_->get_dtype()};
        size_t hidden_size = model_config_->get<size_t>("hidden_size");
        size_t intermediate_size = model_config_->get<size_t>("intermediate_size");
        double rms_norm_eps = model_config_->get<double>("rms_norm_eps");

        // Initialize layer normalization layers
        input_layernorm_ = this->register_module<infinicore::nn::RMSNorm>("input_layernorm", hidden_size, rms_norm_eps, dtype, device);
        post_attention_layernorm_ = this->register_module<infinicore::nn::RMSNorm>("post_attention_layernorm", hidden_size, rms_norm_eps, dtype, device);

        // Initialize attention and MLP modules
        mlp_ = this->register_module<MLP>("mlp", model_config_, device);

        std::vector<std::string> mixer_types = model_config_->get<std::vector<std::string>>("mixer_types");
        std::string mixer_type = mixer_types[layer_idx];

        if ("minicpm4" == mixer_type) {
            self_attn_ = std::make_shared<Attention>(this->register_module<InfLLMv2Attention>("self_attn", model_config_, layer_idx, device));
        } else if ("lightning" == mixer_type || "lightning_attn" == mixer_type || "lightning-attn" == mixer_type) {
            self_attn_ = std::make_shared<Attention>(this->register_module<LightningAttention>("self_attn", model_config_, layer_idx, device));
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
            infinicore::Tensor &residual) {

        // 1. Attention layer normalization
        input_layernorm_->forward_inplace(hidden_states, residual);

        // 2. Self-attention
        hidden_states = std::visit(
            [&](auto &attn_ptr) { return attn_ptr->forward(hidden_states); }, *self_attn_);

        // 3. Post-attention layer normalization
        post_attention_layernorm_->forward_inplace(hidden_states, residual);

        // 4. MLP
        hidden_states = mlp_->forward(hidden_states);
        return std::make_tuple(hidden_states, residual);
    }

    // Compatibility overload for TemplateModel::forward_naive: expects
    // DecoderLayer::forward(hidden_states, input, kv_cache) -> hidden_states
    infinicore::Tensor forward(infinicore::Tensor &hidden_states) {
        auto residual = hidden_states;
        auto [new_hidden, new_residual] = forward(hidden_states, residual);
        hidden_states = new_hidden;
        (void)new_residual;
        return hidden_states;
    }

    /**
     * @brief Get the layer index
     */
    size_t layer_idx() const { return layer_idx_; }

    void set_rotary_emb(const std::shared_ptr<infinicore::nn::RoPE> &rotary_emb) {
        if (self_attn_) {
            std::visit([&](auto &attn_ptr) { attn_ptr->set_rotary_emb(rotary_emb); }, *self_attn_);
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

} // namespace infinilm::models::minicpm_sala
