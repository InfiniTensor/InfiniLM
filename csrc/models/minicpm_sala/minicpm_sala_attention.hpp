#pragma once

#include "../../layers/common_modules.hpp"

namespace infinilm::layers::attention {
class AttentionLayer;
}

namespace infinilm::models::minicpm_sala {

class AttentionBase : public infinicore::nn::Module {
protected:
    AttentionBase(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                  size_t num_attention_heads,
                  size_t num_key_value_heads,
                  size_t layer_idx,
                  const infinicore::Device &device);

public:
    size_t layer_idx() const { return layer_idx_; }
    size_t num_heads() const { return num_attention_heads_; }
    size_t num_kv_heads() const { return num_key_value_heads_; }
    size_t head_dim() const { return head_dim_; }
    size_t hidden_size() const { return hidden_size_; }

protected:
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> q_proj_;
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> k_proj_;
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> v_proj_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> o_proj_;

    std::shared_ptr<infinilm::layers::attention::AttentionLayer> attn_;
    ::infinilm::backends::AttentionBackend attention_backend_;
    std::shared_ptr<infinicore::nn::RoPE> rotary_emb_;

    size_t layer_idx_;
    size_t hidden_size_;
    size_t num_attention_heads_;
    size_t num_key_value_heads_;
    size_t head_dim_;
    bool use_bias_;
    bool use_output_bias_;

    // For off-line kv cache quantization
    infinicore::nn::Parameter kv_cache_k_scale_;
    infinicore::nn::Parameter kv_cache_v_scale_;
};

/**
 * @brief InfLLMv2 attention with optional output gate
 */
class InfLLMv2Attention : public AttentionBase {
public:
    InfLLMv2Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                      size_t layer_idx,
                      const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               const infinicore::Tensor &hidden_states) const;

protected:
    bool use_output_gate_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> o_gate_;
};

/**
 * @brief Lightning attention with optional output norm and gate
 */
class LightningAttention : public AttentionBase {
public:
    LightningAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                       size_t layer_idx,
                       const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               const infinicore::Tensor &hidden_states) const;

protected:
    bool qk_norm_;
    bool use_output_norm_;
    bool use_output_gate_;
    std::shared_ptr<infinicore::nn::RMSNorm> q_norm_;
    std::shared_ptr<infinicore::nn::RMSNorm> k_norm_;
    std::shared_ptr<infinicore::nn::RMSNorm> o_norm_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> z_proj_;
};

} // namespace infinilm::models::minicpm_sala
