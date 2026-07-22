#pragma once

#include "../../backends/attention_backends.hpp"
#include "../../global_state/piecewise_prefill_state.hpp"
#include "../../layers/common_modules.hpp"
#include "../../layers/linear/fused_linear.hpp"
#include "infinicore/ops/inductor_segment.hpp"

namespace infinilm::models::minicpm5_moe {

/**
 * Gated MiniCPM5-MoE attention (q_proj out = 2 * H * D).
 * Implements the piecewise prefill hooks expected by TextModel / PiecewiseTextCausalLM.
 */
class MiniCPM5MoeAttention : public infinicore::nn::Module {
public:
    MiniCPM5MoeAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         size_t layer_idx,
                         const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               const infinicore::Tensor &hidden_states) const;

    void forward_pre_attn_piecewise(const infinicore::Tensor &positions,
                                    const infinicore::Tensor &hidden_states,
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

    infinicore::op::inductor_segment_impl::PreAttnExternalWeightTensors
    pre_attn_external_weights() const;

    void process_fused_weights_after_loading() {
        qkv_proj_->process_weights_after_loading();
    }

    size_t layer_idx() const { return layer_idx_; }

protected:
    INFINICORE_NN_MODULE(infinilm::layers::linear::QKVParallelLinear, qkv_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::RowParallelLinear, o_proj);

    size_t layer_idx_{0};
    size_t hidden_size_{0};
    size_t head_dim_{0};
    size_t num_attention_heads_{0};
    size_t num_key_value_heads_{0};
    bool use_gated_attention_{false};

    ::infinilm::backends::AttentionBackend attention_backend_;
    std::shared_ptr<infinicore::nn::RoPE> rotary_emb_;
    std::shared_ptr<infinilm::layers::attention::AttentionLayer> attn_;
    INFINICORE_NN_PARAMETER(kv_cache_k_scale);
    INFINICORE_NN_PARAMETER(kv_cache_v_scale);

    /// Stable per-layer gate buffers (avoid CG HostOp / free-list aliasing of
    /// per-forward ``contiguous()`` / ``sigmoid()`` temporaries).
    mutable infinicore::Tensor gate_score_cache_;
    mutable infinicore::Tensor gate_sigmoid_buf_;
};

} // namespace infinilm::models::minicpm5_moe
