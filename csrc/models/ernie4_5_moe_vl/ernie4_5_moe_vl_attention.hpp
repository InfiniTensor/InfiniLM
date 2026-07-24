#pragma once

#include "../../layers/common_modules.hpp"

#include <tuple>
#include <vector>

namespace infinilm::models::ernie4_5_moe_vl {

// Text-side self-attention for ERNIE-4.5-VL-MoE.
//
// Differences vs. the generic `infinilm::layers::attention::Attention`:
//   - No q_norm / k_norm (ERNIE-4.5 attention has no QK-RMSNorm).
//   - 3D RoPE (mrope): position_ids carry [time, height, width]; rotation uses
//     `rope_scaling.mrope_section = [22, 22, 20]`.
//   - use_bias = false for all projections (config: "use_bias": false).
//
// GQA: num_attention_heads = 20, num_key_value_heads = 4, head_dim = 128.
class Ernie4_5_VLMoeAttention : public infinicore::nn::Module {
public:
    Ernie4_5_VLMoeAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                            size_t layer_idx,
                            const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               const infinicore::Tensor &hidden_states) const;

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

    // Build per-token 3D-mrope sin/cos tables [seq, head_dim/2] plus a [seq]
    // position index (arange), so op::rope can be reused: each table row already
    // encodes the multi-axis (time,height,width) rotation for that token.
    std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor>
    build_mrope_(const infinicore::Tensor &position_ids,
                 const infinicore::DataType &dtype,
                 const infinicore::Device &device) const;

protected:
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

    // 3D mrope (rope_scaling.mrope_section, e.g. [22,22,20] summing to head_dim/2).
    // Empty -> plain 1D rope via rotary_emb_.
    std::vector<size_t> mrope_section_;
    double rope_theta_{0.0};
    infinicore::nn::RoPE::Algo rope_algo_{infinicore::nn::RoPE::Algo::GPT_NEOX};

    // For off-line kv cache quantization.
    INFINICORE_NN_PARAMETER(kv_cache_k_scale);
    INFINICORE_NN_PARAMETER(kv_cache_v_scale);
};

} // namespace infinilm::models::ernie4_5_moe_vl
