#pragma once

#include "../../backends/attention_backends.hpp"
#include "../../config/model_config.hpp"
#include "../../global_state/global_state.hpp"
#include "../../global_state/piecewise_prefill_state.hpp"
#include "../linear/linear.hpp"
#include "backends/attention_layer.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/tensor.hpp"
#include <memory>

namespace infinilm::layers::attention {
class Attention : public infinicore::nn::Module {
public:
    Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
              size_t layer_idx,
              const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               const infinicore::Tensor &hidden_states) const;

    /// Piecewise prefill: QKV + copy to staging (RoPE applied eagerly via apply_rope).
    void forward_pre_attn_piecewise(const infinicore::Tensor &positions,
                                    const infinicore::Tensor &hidden_states,
                                    global_state::PiecewiseLayerStaging &staging) const;

    /// Piecewise eager segment: refresh Q/K/V staging after pre_attn graph replay.
    void forward_pre_attn_piecewise_fill_staging(const infinicore::Tensor &hidden_states,
                                                global_state::PiecewiseLayerStaging &staging) const;

    /// Piecewise prefill: eager RoPE on staging Q/K (outside pre_attn graph).
    void forward_pre_attn_piecewise_apply_rope(const infinicore::Tensor &position_ids,
                                               global_state::PiecewiseLayerStaging &staging) const;

    /// Piecewise prefill: KV cache + varlen attention (eager, outside graph).
    void forward_eager_attn_piecewise(const infinicore::Tensor &positions,
                                      global_state::PiecewiseLayerStaging &staging) const;

    /// Piecewise prefill: O-proj into ``hidden_states`` persistent buffer (stable at capture/replay).
    void forward_post_attn_piecewise_into(infinicore::Tensor &hidden_states,
                                          global_state::PiecewiseLayerStaging &staging) const;

    /// Piecewise graph segment: O-proj matmul + in-graph allreduce (vLLM order).
    void forward_post_attn_piecewise_graph_into(infinicore::Tensor &hidden_states,
                                                global_state::PiecewiseLayerStaging &staging) const;

    /// Piecewise eager segment: O-proj allreduce after graph replay.
    void forward_post_attn_piecewise_allreduce_into(infinicore::Tensor &hidden_states,
                                                    global_state::PiecewiseLayerStaging &staging) const;

    /// Recompute O-proj matmul into ``ar_staging`` from ``staging.attn_output`` (post-graph refresh).
    void forward_post_attn_piecewise_matmul_staging(infinicore::Tensor &hidden_states,
                                                    global_state::PiecewiseLayerStaging &staging) const;

    void process_fused_weights_after_loading() {
        qkv_proj_->process_weights_after_loading();
    }

    size_t layer_idx() const { return layer_idx_; }
    size_t num_heads() const { return num_attention_heads_; }
    size_t num_kv_heads() const { return num_key_value_heads_; }
    size_t head_dim() const { return head_dim_; }
    size_t hidden_size() const { return hidden_size_; }

private:
    infinicore::Tensor forward_static_(const infinicore::Tensor &positions,
                                       const infinicore::Tensor &hidden_states) const;

    infinicore::Tensor forward_paged_(const infinicore::Tensor &positions,
                                      const infinicore::Tensor &hidden_states) const;

protected:
    std::shared_ptr<infinilm::layers::linear::QKVParallelLinear> qkv_proj_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> o_proj_;
    std::shared_ptr<infinicore::nn::RoPE> rotary_emb_;

    std::shared_ptr<AttentionLayer> attn_;
    ::infinilm::backends::AttentionBackend attention_backend_;
    size_t layer_idx_;
    size_t num_attention_heads_;
    size_t num_key_value_heads_;
    size_t hidden_size_;
    size_t head_dim_;

    // For off-line kv cache quantization
    INFINICORE_NN_PARAMETER(kv_cache_k_scale);
    INFINICORE_NN_PARAMETER(kv_cache_v_scale);
};
void init_kv_cache_quant_params(std::function<void(const std::string &, infinicore::nn::Parameter)> register_fn,
                              const infinicore::Device &device,
                              infinicore::nn::Parameter &kv_cache_k_scale,
                              infinicore::nn::Parameter &kv_cache_v_scale);
} // namespace infinilm::layers::attention
