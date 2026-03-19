#pragma once

#include "../../cache/kv_cache.hpp"
#include "../../config/model_config.hpp"
#include "../../engine/distributed/distributed.hpp"
#include "../../models/infinilm_model.hpp"

#include "infinicore/device.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/tensor.hpp"
#include <memory>

namespace infinilm::layers::attention {
class AttentionLayer;
}

namespace infinilm::models::minicpm_sala {

/**
 * @brief Base class for attention modules, holds shared constructor init and members.
 *
 * StaticAttention and Attention<AttnBackend> inherit from this to reuse
 * Q/K/V projection init, scaling, qk_norm, etc.
 */
class AttentionBase : public infinicore::nn::Module {
protected:
    AttentionBase(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                  size_t num_attention_heads,
                  size_t num_key_value_heads,
                  size_t layer_idx,
                  const infinicore::Device &device,
                  engine::distributed::RankInfo rank_info,
                  ::infinilm::backends::AttentionBackend attention_backend);

public:
    size_t layer_idx() const { return layer_idx_; }
    size_t num_heads() const { return num_attention_heads_; }
    size_t num_kv_heads() const { return num_key_value_heads_; }
    size_t head_dim() const { return head_dim_; }
    size_t hidden_size() const { return hidden_size_; }
    void set_rotary_emb(const std::shared_ptr<infinicore::nn::RoPE> &rotary_emb) { rotary_emb_ = rotary_emb; }

protected:
    INFINICORE_NN_MODULE(infinicore::nn::ColumnParallelLinear, q_proj);
    INFINICORE_NN_MODULE(infinicore::nn::ColumnParallelLinear, k_proj);
    INFINICORE_NN_MODULE(infinicore::nn::ColumnParallelLinear, v_proj);
    INFINICORE_NN_MODULE(infinicore::nn::RowParallelLinear, o_proj);

    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, q_norm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, k_norm);
    engine::distributed::RankInfo rank_info_;

    std::shared_ptr<infinilm::layers::attention::AttentionLayer> attn_;

    std::shared_ptr<infinicore::nn::RoPE> rotary_emb_;
    std::shared_ptr<infinilm::config::ModelConfig> model_config_ = std::make_shared<infinilm::config::ModelConfig>();
    size_t layer_idx_;
    size_t hidden_size_;
    size_t num_attention_heads_;
    size_t num_key_value_heads_;
    size_t head_dim_;
    size_t kv_dim_;
    bool use_bias_;
    bool use_output_bias_;
    size_t max_position_embeddings_;
    float scaling_;
};

/**
 * @brief InfLLMv2/MiniCPM4-style attention with optional output gate
 */
class InfLLMv2Attention : public AttentionBase {
public:
    InfLLMv2Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                      size_t layer_idx,
                      const infinicore::Device &device,
                      engine::distributed::RankInfo rank_info,
                      ::infinilm::backends::AttentionBackend attention_backend);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinilm::InfinilmModel::Input &input,
                               std::shared_ptr<infinilm::cache::Cache> kv_cache) const;

protected:
    const bool use_output_gate_;
    INFINICORE_NN_MODULE(infinicore::nn::Linear, o_gate);
};

/**
 * @brief Lightning-style attention with optional output norm and gate
 */
class LightningAttention : public AttentionBase {
public:
    LightningAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                       size_t layer_idx,
                       const infinicore::Device &device,
                       engine::distributed::RankInfo rank_info,
                       ::infinilm::backends::AttentionBackend attention_backend);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinilm::InfinilmModel::Input &input,
                               std::shared_ptr<infinilm::cache::Cache> kv_cache) const;

protected:
    const bool qk_norm_;
    const bool use_output_norm_;
    const bool use_output_gate_;
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, o_norm);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, z_proj);
};

} // namespace infinilm::models::minicpm_sala