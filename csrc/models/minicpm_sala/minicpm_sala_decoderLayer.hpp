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

using MiniCPMMLP = infinilm::layers::MLP;
using MiniCPMSALAAttention = std::variant<std::shared_ptr<InfLLMv2Attention>, std::shared_ptr<LightningAttention>>;

class MiniCPMSALADecoderLayer : public infinicore::nn::Module {
public:
    MiniCPMSALADecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                            size_t layer_idx,
                            const infinicore::Device &device)
        : layer_idx_(layer_idx) {
        const auto &dtype{model_config->get_dtype()};
        size_t hidden_size = model_config->get<size_t>("hidden_size");
        size_t intermediate_size = model_config->get<size_t>("intermediate_size");
        double rms_norm_eps = model_config->get<double>("rms_norm_eps");

        // Initialize layer normalization layers
        INFINICORE_NN_MODULE_INIT(input_layernorm, hidden_size, rms_norm_eps, dtype, device);
        INFINICORE_NN_MODULE_INIT(post_attention_layernorm, hidden_size, rms_norm_eps, dtype, device);

        // Initialize attention and MLP modules
        INFINICORE_NN_MODULE_INIT(mlp, model_config, device);

        std::vector<std::string> mixer_types = model_config->get<std::vector<std::string>>("mixer_types");
        std::string mixer_type = mixer_types[layer_idx];
        if ("minicpm4" == mixer_type) {
            self_attn_ = std::make_shared<MiniCPMSALAAttention>(this->register_module<InfLLMv2Attention>("self_attn", model_config, layer_idx, device));
        } else if ("lightning" == mixer_type || "lightning_attn" == mixer_type || "lightning-attn" == mixer_type) {
            self_attn_ = std::make_shared<MiniCPMSALAAttention>(this->register_module<LightningAttention>("self_attn", model_config, layer_idx, device));
        } else {
            throw std::runtime_error("infinilm::models::minicpm_sala::MiniCPMSALADecoderLayer: unsupported mixer_type '" + mixer_type + "' for layer " + std::to_string(layer_idx));
        }
    }

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(infinicore::Tensor &hidden_states, infinicore::Tensor &residual) {

        input_layernorm_->forward_inplace(hidden_states, residual);
        hidden_states = std::visit(
            [&](auto &attn_ptr) { return attn_ptr->forward(hidden_states); }, *self_attn_);

        post_attention_layernorm_->forward_inplace(hidden_states, residual);
        hidden_states = mlp_->forward(hidden_states);
        return std::make_tuple(hidden_states, residual);
    }

    infinicore::Tensor forward(infinicore::Tensor &hidden_states) {
        auto residual = hidden_states;

        hidden_states = input_layernorm_->forward(hidden_states);
        hidden_states = std::visit(
            [&](auto &attn_ptr) { return attn_ptr->forward(hidden_states); }, *self_attn_);

        hidden_states = infinicore::op::add(residual, hidden_states);

        residual = hidden_states;
        hidden_states = post_attention_layernorm_->forward(hidden_states);
        hidden_states = mlp_->forward(hidden_states);
        hidden_states = infinicore::op::add(residual, hidden_states);
        return hidden_states;
    }

    void set_rotary_emb(const std::shared_ptr<infinicore::nn::RoPE> &rotary_emb) {
        if (self_attn_) {
            std::visit([&](auto &attn_ptr) { attn_ptr->set_rotary_emb(rotary_emb); }, *self_attn_);
        }
    }

protected:
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(MiniCPMSALAAttention, self_attn);
    INFINICORE_NN_MODULE(MiniCPMMLP, mlp);

    size_t layer_idx_; // Layer index for cache management and debugging
};

} // namespace infinilm::models::minicpm_sala
