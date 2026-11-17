#pragma once

#include "../../layers/common_modules.hpp"
#include "../../global_state/piecewise_prefill_state.hpp"

namespace infinilm::models::minicpm5_moe {

/**
 * GQA attention with optional gated output (doubled q_proj, per-head [q|gate]).
 * Matches HF MiniCPM5MoEAttention / vllm_minicpm5 gated path (no q/k RMSNorm).
 */
class MiniCPM5MoEAttention : public infinicore::nn::Module {
public:
    MiniCPM5MoEAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         size_t layer_idx,
                         const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               const infinicore::Tensor &hidden_states) const;


    void forward_pre_attn_piecewise(const infinicore::Tensor &positions,
                                    const infinicore::Tensor &hidden_states,
                                    global_state::PiecewiseLayerStaging &staging) const;

    void forward_pre_attn_qkv_piecewise(const infinicore::Tensor &positions,
                                        const infinicore::Tensor &hidden_states,
                                        global_state::PiecewiseLayerStaging &staging) const;

    void forward_pre_attn_rope_piecewise(const infinicore::Tensor &positions,
                                         global_state::PiecewiseLayerStaging &staging) const;

    void forward_eager_attn_piecewise(const infinicore::Tensor &positions,
                                      global_state::PiecewiseLayerStaging &staging) const;

    void forward_post_attn_piecewise_into(infinicore::Tensor &hidden_states,
                                          global_state::PiecewiseLayerStaging &staging) const;

    void forward_post_attn_piecewise_graph_into(infinicore::Tensor &hidden_states,
                                                global_state::PiecewiseLayerStaging &staging) const;

    void forward_post_attn_piecewise_cg_into(infinicore::Tensor &hidden_states,
                                             global_state::PiecewiseLayerStaging &staging) const;

    void forward_post_attn_piecewise_allreduce_into(infinicore::Tensor &hidden_states,
                                                    global_state::PiecewiseLayerStaging &staging) const;

    void process_weights_after_loading() override {
        qkv_proj_->process_weights_after_loading();
    }

    void reset_runtime_state() const override {
        qkv_proj_->reset_runtime_state();
    }

    size_t layer_idx() const { return layer_idx_; }
    size_t num_heads() const { return num_attention_heads_; }
    size_t num_kv_heads() const { return num_key_value_heads_; }
    size_t head_dim() const { return head_dim_; }
    size_t hidden_size() const { return hidden_size_; }

protected:
    infinicore::Tensor forward_static_(const infinicore::Tensor &position_ids,
                                       const infinicore::Tensor &hidden_states) const;
    infinicore::Tensor forward_paged_(const infinicore::Tensor &position_ids,
                                      const infinicore::Tensor &hidden_states) const;

    std::shared_ptr<infinilm::layers::linear::QKVParallelLinear> qkv_proj_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> o_proj_;
    std::shared_ptr<infinicore::nn::RoPE> rotary_emb_;

    std::shared_ptr<infinilm::layers::attention::AttentionLayer> attn_;
    ::infinilm::backends::AttentionBackend attention_backend_;
    size_t layer_idx_;
    size_t num_attention_heads_;
    size_t num_key_value_heads_;
    size_t hidden_size_;
    size_t head_dim_;
    bool use_gated_attention_{true};

    INFINICORE_NN_PARAMETER(kv_cache_k_scale);
    INFINICORE_NN_PARAMETER(kv_cache_v_scale);

    mutable infinicore::Tensor gate_score_cache_;
    mutable infinicore::Tensor gate_score_write_view_;
    mutable infinicore::Tensor gate_sigmoid_buf_;
};


} // namespace infinilm::models::minicpm5_moe
