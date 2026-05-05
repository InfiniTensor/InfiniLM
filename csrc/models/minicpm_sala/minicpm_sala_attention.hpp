#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/rotary_embedding/rotary_embedding.hpp"

#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <string>

namespace infinilm::models::minicpm_sala {

class MiniCPMSALAAttentionBase : public infinicore::nn::Module {
public:
    virtual infinicore::Tensor forward(const infinicore::Tensor &position_ids,
                                       const infinicore::Tensor &hidden_states) const = 0;
    virtual void reset_state() = 0;
    virtual ~MiniCPMSALAAttentionBase() = default;
};

// Lightning attention path (Simple GLA). Parameter names align with HF:
//   model.layers.N.self_attn.{q_proj,k_proj,v_proj,o_proj,q_norm,k_norm,o_norm,z_proj,...}
class MiniCPMSALALightningAttention : public MiniCPMSALAAttentionBase {
public:
    MiniCPMSALALightningAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                  const infinicore::Device &device,
                                  size_t layer_idx);

    // Match `infinilm::layers::attention::Attention` API: metadata is pulled from
    // `global_state::get_forward_context().attn_metadata`.
    infinicore::Tensor forward(const infinicore::Tensor &position_ids,
                               const infinicore::Tensor &hidden_states) const override;

    void reset_state() override;

protected:
    // Projections (HF-aligned naming)
    INFINICORE_NN_MODULE(infinicore::nn::Linear, q_proj);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, k_proj);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, v_proj);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, o_proj);

    // Optional (Lightning layers): q_norm/k_norm/o_norm + z_proj
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, q_norm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, k_norm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, o_norm);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, z_proj);

    std::shared_ptr<infinicore::nn::RoPE> rotary_emb_;

    size_t layer_idx_;
    size_t num_attention_heads_;
    size_t num_key_value_heads_;
    size_t head_dim_;
    float scaling_;

    bool use_qk_norm_ = false;
    bool use_output_gate_ = false;
    bool use_output_norm_ = false;
    bool use_rope_ = false;

    // Lightning layers only: per-head log-decay for Simple GLA (HF _build_slope_tensor * -1).
    infinicore::Tensor g_gamma_;

    // Lightning layers only: recurrent state for fast decode.
    // Shape: [B, H, D, D] float32. Tracks how many KV tokens are folded into the state.
    mutable infinicore::Tensor gla_state_;
    mutable size_t gla_state_cached_len_ = 0;
    mutable bool gla_state_valid_ = false;
};

// Sparse attention path (`mixer_type=="minicpm4"`) using InfLLM-v2 operators.
// Parameter names align with HF:
//   model.layers.N.self_attn.{q_proj,k_proj,v_proj,o_proj,o_gate,...}
class MiniCPMSALAMinicpm4Attention : public MiniCPMSALAAttentionBase {
public:
    MiniCPMSALAMinicpm4Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                 const infinicore::Device &device,
                                 size_t layer_idx);

    infinicore::Tensor forward(const infinicore::Tensor &position_ids,
                               const infinicore::Tensor &hidden_states) const override;

    void reset_state() override;

protected:
    INFINICORE_NN_MODULE(infinicore::nn::Linear, q_proj);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, k_proj);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, v_proj);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, o_proj);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, o_gate);

    size_t layer_idx_;
    size_t num_attention_heads_;
    size_t num_key_value_heads_;
    size_t head_dim_;
    float scaling_;

    // InfLLM-v2 local-window masking plumbing.
    int infllmv2_window_left_ = -1;
    bool use_local_window_ = false;
};

} // namespace infinilm::models::minicpm_sala
