#pragma once

#include "../../config/model_config.hpp"
#include "../../global_state/piecewise_prefill_state.hpp"
#include "infinicore/device.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/tensor.hpp"
#include <memory>
#include <tuple>
namespace infinilm::layers::causal_lm_templates {

/**
 * @brief Generic Text decoder layer (transformer block) class.
 *
 * @tparam Attention: The attention module type (e.g., Qwen3Attention)
 * @tparam MLP: The MLP module type (e.g., Qwen3MLP)
 */
template <typename Attention, typename MLP>
class TextDecoderLayer : public infinicore::nn::Module {
public:
    TextDecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                     size_t layer_idx,
                     const infinicore::Device &device)
        : layer_idx_(layer_idx) {

        const auto &dtype{model_config->get_dtype()};
        size_t hidden_size = model_config->get<size_t>("hidden_size");
        double rms_norm_eps = model_config->get<double>("rms_norm_eps");

        input_layernorm_ = this->register_module<infinicore::nn::RMSNorm>("input_layernorm", hidden_size, rms_norm_eps, dtype, device);
        post_attention_layernorm_ = this->register_module<infinicore::nn::RMSNorm>("post_attention_layernorm", hidden_size, rms_norm_eps, dtype, device);

        self_attn_ = this->register_module<Attention>("self_attn", model_config, layer_idx, device);
        mlp_ = this->register_module<MLP>("mlp", model_config, device);
    }

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(const infinicore::Tensor &positions,
                                                               infinicore::Tensor &hidden_states,
                                                               infinicore::Tensor &residual) {
        input_layernorm_->forward_inplace(hidden_states, residual);
        hidden_states = self_attn_->forward(positions, hidden_states);
        post_attention_layernorm_->forward_inplace(hidden_states, residual);
        hidden_states = mlp_->forward(hidden_states);
        return std::make_tuple(hidden_states, residual);
    }

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               infinicore::Tensor &hidden_states) {
        auto residual = hidden_states;
        hidden_states = input_layernorm_->forward(hidden_states);
        hidden_states = self_attn_->forward(positions, hidden_states);
        hidden_states = infinicore::op::add(residual, hidden_states);

        residual = hidden_states;
        hidden_states = post_attention_layernorm_->forward(hidden_states);
        hidden_states = mlp_->forward(hidden_states);
        hidden_states = infinicore::op::add(residual, hidden_states);
        return hidden_states;
    }

    size_t layer_idx() const { return layer_idx_; }

    void piecewise_pre_attn(const infinicore::Tensor &positions,
                            infinicore::Tensor &hidden_states,
                            infinicore::Tensor &residual,
                            global_state::PiecewiseLayerStaging &staging) const {
        input_layernorm_->forward_inplace(hidden_states, residual);
        self_attn_->forward_pre_attn_piecewise(positions, hidden_states, staging);
    }

    void piecewise_eager_attn(const infinicore::Tensor &positions,
                              global_state::PiecewiseLayerStaging &staging) const {
        self_attn_->forward_eager_attn_piecewise(positions, staging);
    }

    void piecewise_post_attn(infinicore::Tensor &hidden_states,
                             infinicore::Tensor &residual,
                             global_state::PiecewiseLayerStaging &staging) const {
        self_attn_->forward_post_attn_piecewise_into(hidden_states, staging);
        post_attention_layernorm_->forward_inplace(hidden_states, residual);
        auto mlp_out = mlp_->forward(hidden_states);
        hidden_states->copy_from(mlp_out);
    }

protected:
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(Attention, self_attn);
    INFINICORE_NN_MODULE(MLP, mlp);

    size_t layer_idx_;
};

} // namespace infinilm::layers::causal_lm_templates
