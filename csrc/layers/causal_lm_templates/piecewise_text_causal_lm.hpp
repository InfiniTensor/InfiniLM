#pragma once

#include "text_causal_lm.hpp"

namespace infinilm::layers::causal_lm_templates {

/// TextCausalLM with native piecewise CUDAGraph hooks (TextDecoderLayer-based models).
template <typename Model>
class PiecewiseTextCausalLM : public TextCausalLM<Model> {
public:
    using TextCausalLM<Model>::TextCausalLM;

    bool supports_native_piecewise_prefill() const override { return true; }

    size_t native_piecewise_num_layers() const override {
        return this->model_->num_layers();
    }

    void native_piecewise_embed(const InfinilmModel::Input &input,
                                infinicore::Tensor &hidden_states) const override {
        this->model_->piecewise_embed(input, hidden_states);
    }

    void native_piecewise_pre_attn_layer(size_t layer_idx,
                                         const InfinilmModel::Input &input,
                                         infinicore::Tensor &hidden_states,
                                         infinicore::Tensor &residual) const override {
        this->model_->piecewise_pre_attn_layer(layer_idx, input, hidden_states, residual);
    }

    void native_piecewise_eager_attn_layer(size_t layer_idx,
                                           const InfinilmModel::Input &input) const override {
        this->model_->piecewise_eager_attn_layer(layer_idx, input);
    }

    void native_piecewise_post_attn_layer(size_t layer_idx,
                                          const InfinilmModel::Input &input,
                                          infinicore::Tensor &hidden_states,
                                          infinicore::Tensor &residual) const override {
        this->model_->piecewise_post_attn_layer(layer_idx, input, hidden_states, residual);
    }

    void native_piecewise_lm_head(const InfinilmModel::Input &,
                                  infinicore::Tensor &hidden_states,
                                  infinicore::Tensor &residual,
                                  infinicore::Tensor &logits_out) const override {
        this->model_->piecewise_lm_head(hidden_states, residual);
        auto logits = this->lm_head_->forward(hidden_states);
        logits_out->copy_from(logits);
    }
};

} // namespace infinilm::layers::causal_lm_templates
