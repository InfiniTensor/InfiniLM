#pragma once

#include "text_causal_lm.hpp"

#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"

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

    void native_piecewise_pre_attn_layernorm_layer(size_t layer_idx,
                                                   const InfinilmModel::Input &input,
                                                   infinicore::Tensor &hidden_states,
                                                   infinicore::Tensor &residual) const override {
        this->model_->piecewise_pre_attn_layernorm_layer(layer_idx, input, hidden_states, residual);
    }

    void native_piecewise_pre_attn_rope_layer(size_t layer_idx,
                                              const InfinilmModel::Input &input,
                                              infinicore::Tensor &hidden_states,
                                              infinicore::Tensor &residual) const override {
        this->model_->piecewise_pre_attn_rope_layer(layer_idx, input, hidden_states, residual);
    }

    void native_piecewise_pre_attn_staging_layer(size_t layer_idx,
                                                 const InfinilmModel::Input &input,
                                                 infinicore::Tensor &hidden_states,
                                                 infinicore::Tensor &residual) const override {
        this->model_->piecewise_pre_attn_staging_layer(layer_idx, input, hidden_states, residual);
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

    void native_piecewise_post_attn_graph_layer(size_t layer_idx,
                                                const InfinilmModel::Input &input,
                                                infinicore::Tensor &hidden_states,
                                                infinicore::Tensor &residual) const override {
        this->model_->piecewise_post_attn_graph_layer(layer_idx, input, hidden_states, residual);
    }

    void native_piecewise_post_attn_mlp_graph_layer(size_t layer_idx,
                                                    const InfinilmModel::Input &input,
                                                    infinicore::Tensor &hidden_states,
                                                    infinicore::Tensor &residual) const override {
        this->model_->piecewise_post_attn_mlp_graph_layer(layer_idx, input, hidden_states, residual);
    }

    void native_piecewise_post_attn_allreduce_layer(size_t layer_idx,
                                                    const InfinilmModel::Input &input,
                                                    infinicore::Tensor &hidden_states,
                                                    infinicore::Tensor &residual) const override {
        this->model_->piecewise_post_attn_allreduce_layer(layer_idx, input, hidden_states, residual);
    }

    void native_piecewise_o_proj_staging_layer(size_t layer_idx,
                                               const InfinilmModel::Input &input,
                                               infinicore::Tensor &hidden_states,
                                               infinicore::Tensor &residual) const override {
        this->model_->piecewise_o_proj_staging_layer(layer_idx, input, hidden_states, residual);
    }

    void native_piecewise_lm_head(const InfinilmModel::Input &,
                                  infinicore::Tensor &hidden_states,
                                  infinicore::Tensor &residual,
                                  infinicore::Tensor &logits_out) const override {
        auto &piecewise = infinilm::global_state::get_forward_context().piecewise;
        const size_t bucket = hidden_states->size(1);
        const size_t valid_len = piecewise.valid_seq_len > 0 ? piecewise.valid_seq_len : bucket;
        this->model_->piecewise_lm_head(hidden_states, residual);
        if (valid_len < bucket) {
            auto hidden_narrow = hidden_states->narrow({{1, 0, valid_len}});
            auto logits = this->lm_head_->forward(hidden_narrow);
            logits_out->narrow({{1, 0, valid_len}})->copy_from(logits);
            infinicore::Tensor logits_tail = logits_out->narrow({{1, valid_len, bucket - valid_len}});
            set_zeros(logits_tail);
        } else {
            auto logits = this->lm_head_->forward(hidden_states);
            logits_out->copy_from(logits);
        }
    }
};

} // namespace infinilm::layers::causal_lm_templates
