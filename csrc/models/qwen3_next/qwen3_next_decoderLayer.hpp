#pragma once

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

class Qwen3NextDecoderLayer : public infinicore::nn::Module {
public:
    Qwen3NextDecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
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

        // Initialize MLP
        INFINICORE_NN_MODULE_INIT(mlp, model_config, device);

        // Initialize attention
        const std::vector<std::string> layer_types = model_config->get<std::vector<std::string>>("layer_types");
        layer_type_ = layer_types[layer_idx];
        if ("linear_attention" == layer_type_) {
            INFINICORE_NN_MODULE_INIT(linear_attn, model_config, layer_idx, device);
        } else if ("full_attention" == layer_type_) {
            INFINICORE_NN_MODULE_INIT(self_attn, model_config, layer_idx, device);
        } else {
            throw std::runtime_error("infinilm::models::qwen3_next::Qwen3NextDecoderLayer: unsupported layer_type '" +
                                    layer_type_ + "' for layer " + std::to_string(layer_idx));
        }
    }

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(infinicore::Tensor &hidden_states,
                                                               infinicore::Tensor &residual) {

        input_layernorm_->forward_inplace(hidden_states, residual);

        if ("linear_attention" == layer_type_) {
            hidden_states = linear_attn_->forward(hidden_states);
        } else if ("full_attention" == layer_type_) {
            hidden_states = self_attn_->forward(hidden_states);
        }

        post_attention_layernorm_->forward_inplace(hidden_states, residual);
        hidden_states = mlp_->forward(hidden_states);
        return std::make_tuple(hidden_states, residual);
    }

    infinicore::Tensor forward(infinicore::Tensor &hidden_states) {

        auto residual = hidden_states;

        hidden_states = input_layernorm_->forward(hidden_states);
        if ("linear_attention" == layer_type_) {
            hidden_states = linear_attn_->forward(hidden_states);
        } else if ("full_attention" == layer_type_) {
            hidden_states = self_attn_->forward(hidden_states);
        }
        hidden_states = infinicore::op::add(residual, hidden_states);

        residual = hidden_states;
        hidden_states = post_attention_layernorm_->forward(hidden_states);
        hidden_states = mlp_->forward(hidden_states);
        hidden_states = infinicore::op::add(residual, hidden_states);
        return hidden_states;
    }

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

private:
    size_t layer_idx_;
    std::string layer_type_;
};

} // namespace infinilm::models::qwen3_next
