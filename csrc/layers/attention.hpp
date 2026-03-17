#pragma once

#include "../backends/attention_backends.hpp"
#include "../cache/kv_cache.hpp"
#include "../config/model_config.hpp"
#include "../engine/distributed/distributed.hpp"
#include "../models/infinilm_model.hpp"
#include "../models/llama/llama_config.hpp"
#include "fused_linear.hpp"

#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/tensor.hpp"
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>

namespace infinilm::models::layers {

/**
 * @brief Base class for attention modules, holds shared constructor init and members.
 *
 * StaticAttention and Attention<AttnBackend> inherit from this to reuse
 * Q/K/V projection init, scaling, qk_norm, etc.
 */
class AttentionBase : public infinicore::nn::Module {
protected:
    AttentionBase(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                  const infinicore::Device &device,
                  size_t layer_idx,
                  engine::distributed::RankInfo rank_info);

public:
    size_t layer_idx() const { return layer_idx_; }
    size_t num_heads() const { return num_attention_heads_; }
    size_t num_kv_heads() const { return num_key_value_heads_; }
    size_t head_dim() const { return head_dim_; }
    size_t hidden_size() const { return hidden_size_; }
    void set_rotary_emb(const std::shared_ptr<infinicore::nn::RoPE> &rotary_emb) { rotary_emb_ = rotary_emb; }

protected:
    INFINICORE_NN_MODULE(infinilm::layers::QKVParallelLinear, qkv_proj);
    INFINICORE_NN_MODULE(infinicore::nn::RowParallelLinear, o_proj);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, q_norm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, k_norm);
    engine::distributed::RankInfo rank_info_;

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
    const bool use_qk_norm_;
    size_t max_position_embeddings_;
    float scaling_;
};

/**
 * @brief Template base for paged/Flash attention (CRTP)
 *
 * Implements the attention mechanism with:
 * - Query, Key, Value projections
 * - Output projection
 * - Rotary Position Embeddings (RoPE) applied to Q and K
 * - Support for Grouped Query Attention (GQA)
 * - AttnBackend::attn_calculate() handles actual attention compute (PagedAttention / FlashAttention)
 */
template <typename AttnBackend>
class Attention : public AttentionBase {
public:
    Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
              const infinicore::Device &device,
              size_t layer_idx,
              engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinilm::InfinilmModel::Input &input,
                               std::shared_ptr<infinilm::cache::Cache> kv_cache) const;
};

} // namespace infinilm::models::layers

namespace infinilm::models::layers {
/**
 * @brief Multi-head self-attention module (static KV cache)
 *
 * Implements the attention mechanism with:
 * - Query, Key, Value projections
 * - Output projection
 * - Rotary Position Embeddings (RoPE) applied to Q and K
 * - Support for Grouped Query Attention (GQA)
 * - StaticKVCache only (no PagedKVCache)
 */
class StaticAttention : public AttentionBase {
public:
    StaticAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                    const infinicore::Device &device,
                    size_t layer_idx,
                    engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinilm::InfinilmModel::Input &input,
                               std::shared_ptr<infinilm::cache::Cache> kv_cache) const;
};

class PagedAttention : public Attention<PagedAttention> {
public:
    using Attention<PagedAttention>::Attention;

    infinicore::Tensor attn_calculate(const infinicore::Tensor &q_reshaped,
                                      const infinicore::Tensor &k_reshaped,
                                      const infinicore::Tensor &v_reshaped,
                                      const infinilm::InfinilmModel::Input &input,
                                      std::shared_ptr<infinilm::cache::Cache> kv_cache) const;
};

template class Attention<PagedAttention>;

class FlashAttention : public Attention<FlashAttention> {
public:
    using Attention<FlashAttention>::Attention;

    infinicore::Tensor attn_calculate(const infinicore::Tensor &q_reshaped,
                                      const infinicore::Tensor &k_reshaped,
                                      const infinicore::Tensor &v_reshaped,
                                      const infinilm::InfinilmModel::Input &input,
                                      std::shared_ptr<infinilm::cache::Cache> kv_cache) const;
};

template class Attention<FlashAttention>;

class FlashInferAttention : public Attention<FlashInferAttention> {
public:
    using Attention<FlashInferAttention>::Attention;
    inline infinicore::Tensor attn_calculate(const infinicore::Tensor &q_reshaped,
                                             const infinicore::Tensor &k_reshaped,
                                             const infinicore::Tensor &v_reshaped,
                                             const infinilm::InfinilmModel::Input &input,
                                             std::shared_ptr<infinilm::cache::Cache> kv_cache) const;
};

template class Attention<FlashInferAttention>;
} // namespace infinilm::models::layers
