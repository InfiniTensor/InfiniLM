#pragma once

#include "../../layers/causal_lm_templates/text_decoder_layer.hpp"
#include "infinicore/ops/add.hpp"
#include "infinicore/ops/fused_ffn.hpp"

#include <cstdlib>
#include <string>
#include <tuple>

namespace infinilm::models::fm9g {

// FM9G decoder layer that may substitute the post-attention rms_norm + MLP
// block with the InfiniCore fused-FFN op.
//
// The substitution is gated per `forward()` call by `INFINILM_USE_FUSED_FFN`
// so benchmarks can interleave fused and non-fused passes within one process.
// When MuP scaling on `down_proj` is active (alpha != 1.0), the per-op path
// is taken to preserve the multiplier the fused kernel does not model.
//
// The fused kernel accepts only rank-2 `[ntok, hidden]` tensors, so the call
// site views `hidden_states` to 2-D and back.
template <typename Attention, typename MLP>
class FM9GFusedDecoderLayer
    : public infinilm::layers::causal_lm_templates::TextDecoderLayer<Attention, MLP> {
    using Base = infinilm::layers::causal_lm_templates::TextDecoderLayer<Attention, MLP>;

public:
    FM9GFusedDecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                          size_t layer_idx,
                          const infinicore::Device &device)
        : Base(model_config, layer_idx, device),
          rms_norm_eps_(static_cast<float>(model_config->get<double>("rms_norm_eps"))) {}

    std::tuple<infinicore::Tensor, infinicore::Tensor>
    forward(const infinicore::Tensor &positions,
            infinicore::Tensor &hidden_states,
            infinicore::Tensor &residual) {
        this->input_layernorm_->forward_inplace(hidden_states, residual);
        hidden_states = this->self_attn_->forward(positions, hidden_states);

        if (use_fused_ffn()) {
            residual = infinicore::op::add(residual, hidden_states);
            auto fused_in_2d = as_2d(residual);
            auto fused_out_2d = infinicore::op::fused_ffn(
                fused_in_2d, std::nullopt,
                this->post_attention_layernorm_->weight(),
                this->mlp_->gate_up_weight(),
                this->mlp_->down_weight(),
                rms_norm_eps_);
            hidden_states = fused_out_2d->view(residual->shape());
        } else {
            this->post_attention_layernorm_->forward_inplace(hidden_states, residual);
            hidden_states = this->mlp_->forward(hidden_states);
        }
        return std::make_tuple(hidden_states, residual);
    }

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               infinicore::Tensor &hidden_states) {
        auto residual = hidden_states;
        hidden_states = this->input_layernorm_->forward(hidden_states);
        hidden_states = this->self_attn_->forward(positions, hidden_states);
        hidden_states = infinicore::op::add(residual, hidden_states);

        if (use_fused_ffn()) {
            const auto orig_shape = hidden_states->shape();
            auto fused_in_2d = as_2d(hidden_states);
            auto fused_residual_2d = as_2d(hidden_states);
            auto fused_out_2d = infinicore::op::fused_ffn(
                fused_in_2d, fused_residual_2d,
                this->post_attention_layernorm_->weight(),
                this->mlp_->gate_up_weight(),
                this->mlp_->down_weight(),
                rms_norm_eps_);
            hidden_states = fused_out_2d->view(orig_shape);
        } else {
            residual = hidden_states;
            hidden_states = this->post_attention_layernorm_->forward(hidden_states);
            hidden_states = this->mlp_->forward(hidden_states);
            hidden_states = infinicore::op::add(residual, hidden_states);
        }
        return hidden_states;
    }

private:
    bool use_fused_ffn() const {
        if (this->mlp_->down_alpha() != 1.0f) {
            return false;
        }
        const char *env = std::getenv("INFINILM_USE_FUSED_FFN");
        return env != nullptr && std::string(env) == "1";
    }

    static infinicore::Tensor as_2d(const infinicore::Tensor &t) {
        const auto &shape = t->shape();
        size_t hidden = shape.back();
        size_t ntok = 1;
        for (size_t i = 0; i + 1 < shape.size(); ++i) {
            ntok *= shape[i];
        }
        return t->view({ntok, hidden});
    }

    float rms_norm_eps_;
};

} // namespace infinilm::models::fm9g
