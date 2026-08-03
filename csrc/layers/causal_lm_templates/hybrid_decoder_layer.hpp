#pragma once

#include "../../config/model_config.hpp"
#include "infinicore/device.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace infinilm::layers::causal_lm_templates {

template <typename Attention, typename LinearAttention, typename MLP>
class HybridDecoderLayer : public infinicore::nn::Module {
public:
    HybridDecoderLayer(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        size_t layer_idx,
        const infinicore::Device &device)
        : layer_idx_(layer_idx) {
        const auto &dtype = model_config->get_dtype();
        const size_t hidden_size = model_config->get<size_t>("hidden_size");
        const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

        input_layernorm_ = this->register_module<infinicore::nn::RMSNorm>(
            "input_layernorm", hidden_size, rms_norm_eps, dtype, device);
        post_attention_layernorm_ = this->register_module<infinicore::nn::RMSNorm>(
            "post_attention_layernorm", hidden_size, rms_norm_eps, dtype, device);
        mlp_ = register_mlp(model_config, layer_idx, device);

        const auto layer_types = model_config->get<std::vector<std::string>>("layer_types");
        const std::string &layer_type = layer_types.at(layer_idx);
        if (layer_type == "linear_attention") {
            is_linear_attention_ = true;
            linear_attn_ = this->register_module<LinearAttention>(
                "linear_attn", model_config, layer_idx, device);
        } else if (layer_type == "full_attention") {
            self_attn_ = this->register_module<Attention>(
                "self_attn", model_config, layer_idx, device);
        } else {
            throw std::runtime_error(
                "HybridDecoderLayer: unsupported layer_type '" + layer_type
                + "' for layer " + std::to_string(layer_idx));
        }
    }

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(
        const infinicore::Tensor &positions,
        infinicore::Tensor &hidden_states,
        infinicore::Tensor &residual) {
        input_layernorm_->forward_inplace(hidden_states, residual);
        hidden_states = forward_mixer(positions, hidden_states);
        post_attention_layernorm_->forward_inplace(hidden_states, residual);
        hidden_states = mlp_->forward(hidden_states);
        return std::make_tuple(hidden_states, residual);
    }

    infinicore::Tensor forward(
        const infinicore::Tensor &positions,
        infinicore::Tensor &hidden_states) {
        auto residual = hidden_states;
        hidden_states = input_layernorm_->forward(hidden_states);
        hidden_states = forward_mixer(positions, hidden_states);
        hidden_states = infinicore::op::add(residual, hidden_states);

        residual = hidden_states;
        hidden_states = post_attention_layernorm_->forward(hidden_states);
        hidden_states = mlp_->forward(hidden_states);
        return infinicore::op::add(residual, hidden_states);
    }

    size_t layer_idx() const { return layer_idx_; }

protected:
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(Attention, self_attn);
    INFINICORE_NN_MODULE(LinearAttention, linear_attn);
    INFINICORE_NN_MODULE(MLP, mlp);

private:
    infinicore::Tensor forward_mixer(
        const infinicore::Tensor &positions,
        infinicore::Tensor &hidden_states) const {
        if (is_linear_attention_) {
            return linear_attn_->forward(hidden_states);
        }
        return self_attn_->forward(positions, hidden_states);
    }

    std::shared_ptr<MLP> register_mlp(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        size_t layer_idx,
        const infinicore::Device &device) {
        if constexpr (std::is_constructible_v<
                          MLP,
                          std::shared_ptr<infinilm::config::ModelConfig>,
                          size_t,
                          const infinicore::Device &>) {
            return this->register_module<MLP>(
                "mlp", model_config, layer_idx, device);
        } else {
            return this->register_module<MLP>("mlp", model_config, device);
        }
    }

    size_t layer_idx_;
    bool is_linear_attention_{false};
};

} // namespace infinilm::layers::causal_lm_templates
