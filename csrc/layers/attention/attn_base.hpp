#pragma once

#include "../../backends/attention_backends.hpp"
#include "../../cache/kv_cache.hpp"
#include "../../config/model_config.hpp"
#include "../../engine/distributed/distributed.hpp"
#include "../../models/infinilm_model.hpp"
#include "../../models/llama/llama_config.hpp"
#include "../../utils.hpp"
#include "../fused_linear.hpp"

#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/tensor.hpp"
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>

namespace infinilm::models::layers::attention {

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
    const bool qk_norm_;
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
              engine::distributed::RankInfo rank_info = engine::distributed::RankInfo())
        : AttentionBase(std::move(model_config), device, layer_idx, rank_info) {}

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinilm::InfinilmModel::Input &input,
                               std::shared_ptr<infinilm::cache::Cache> kv_cache) const {
        if (!rotary_emb_) {
            throw std::runtime_error("Attention: rotary_emb not configured");
        }

        auto position_ids = input.position_ids.value();

        // Input shape: [batch, seq_len, hidden_size]
        auto hidden_states_mutable = hidden_states;
        auto shape = hidden_states->shape();
        size_t batch_size = shape[0];
        size_t seq_len = shape[1];

        // Only support batchsize==1, all requests should be flattened along seqlen dimension
        ASSERT_EQ(batch_size, 1);

        // 1. Project Q, K, V
        auto [q, k, v] = qkv_proj_->forward_split(hidden_states_mutable);

        // 2. Reshape for multi-head attention
        auto q_reshaped = q->view({seq_len, num_attention_heads_, head_dim_});
        auto k_reshaped = k->view({seq_len, num_key_value_heads_, head_dim_});
        auto v_reshaped = v->view({seq_len, num_key_value_heads_, head_dim_});

        if (qk_norm_) {
            q_reshaped = q_norm_->forward(q_reshaped);
            k_reshaped = k_norm_->forward(k_reshaped);
        }

        // 3. Prepare position_ids for RoPE - align with Python pattern
        auto pos_shape = position_ids->shape();
        infinicore::Tensor pos_ids_for_rope = position_ids;
        if (pos_shape.size() == 2) {
            auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
            pos_ids_for_rope = pos_narrowed->view({pos_shape[1]});
        } else if (pos_shape.size() == 1) {
            pos_ids_for_rope = position_ids;
        } else {
            throw std::runtime_error("Unexpected position_ids shape");
        }

        // 4. Apply RoPE to Q and K
        rotary_emb_->forward(q_reshaped, pos_ids_for_rope, true);
        rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);

        // 5. Attn Backend calculate
        auto attn_output = static_cast<const AttnBackend *>(this)->attn_calculate(
            q_reshaped, k_reshaped, v_reshaped, input, std::move(kv_cache));

        // 6. Project output
        attn_output = attn_output->view({1, seq_len, num_attention_heads_ * head_dim_});
        return o_proj_->forward(attn_output);
    }
};

} // namespace infinilm::models::layers::attention
