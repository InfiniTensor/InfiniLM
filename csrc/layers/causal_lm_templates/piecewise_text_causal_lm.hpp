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

    void native_piecewise_pre_attn_qkv_layer(size_t layer_idx,
                                             const InfinilmModel::Input &input,
                                             infinicore::Tensor &hidden_states,
                                             infinicore::Tensor &residual) const override {
        this->model_->piecewise_pre_attn_qkv_layer(layer_idx, input, hidden_states, residual);
    }

    void native_piecewise_pre_attn_rope_layer(size_t layer_idx,
                                              const InfinilmModel::Input &input) const override {
        this->model_->piecewise_pre_attn_rope_layer(layer_idx, input);
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

    void native_piecewise_post_attn_cg_layer(size_t layer_idx,
                                             const InfinilmModel::Input &input,
                                             infinicore::Tensor &hidden_states,
                                             infinicore::Tensor &residual) const override {
        this->model_->piecewise_post_attn_cg_layer(layer_idx, input, hidden_states, residual);
    }

    void native_piecewise_post_attn_decode_cg_layer(size_t layer_idx,
                                                    const InfinilmModel::Input &input,
                                                    infinicore::Tensor &hidden_states,
                                                    infinicore::Tensor &residual) const override {
        this->model_->piecewise_post_attn_decode_cg_layer(layer_idx, input, hidden_states, residual);
    }

    void native_piecewise_eager_moe_layer(size_t layer_idx,
                                          const InfinilmModel::Input &input,
                                          infinicore::Tensor &hidden_states,
                                          infinicore::Tensor &residual) const override {
        this->model_->piecewise_eager_moe_layer(layer_idx, input, hidden_states, residual);
    }

    void native_piecewise_post_attn_graph_layer(size_t layer_idx,
                                                const InfinilmModel::Input &input,
                                                infinicore::Tensor &hidden_states,
                                                infinicore::Tensor &residual) const override {
        this->model_->piecewise_post_attn_graph_layer(layer_idx, input, hidden_states, residual);
    }

    void native_piecewise_post_attn_allreduce_layer(size_t layer_idx,
                                                    const InfinilmModel::Input &input,
                                                    infinicore::Tensor &hidden_states,
                                                    infinicore::Tensor &residual) const override {
        this->model_->piecewise_post_attn_allreduce_layer(layer_idx, input, hidden_states, residual);
    }

    void native_piecewise_lm_head(const InfinilmModel::Input &,
                                  infinicore::Tensor &hidden_states,
                                  infinicore::Tensor &residual,
                                  infinicore::Tensor &logits_out) const override {
        this->model_->piecewise_lm_head(hidden_states, residual);
        auto logits = this->lm_head_->forward(hidden_states);
        // Vocab-parallel non-root keeps a local shard; never blind-copy into a
        // full-vocab bank (or any mismatched shape).
        if (!logits_out || !logits) {
            return;
        }
        if (logits_out->shape() == logits->shape()) {
            logits_out->copy_from(logits);
            return;
        }
        // Rank-aware banks should match; if not, skip rather than throw mid-CG.
        return;
    }
};

} // namespace infinilm::layers::causal_lm_templates
