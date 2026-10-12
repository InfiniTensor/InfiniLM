#pragma once
#include "../../weights/weights.hpp"
#include "../../layers/normalization/rmsnorm.hpp"
#include "../../layers/rotary_embedding/rope.hpp"

#include "../../layers/common_modules.hpp"

namespace infinilm::models::qwen3_next {

class Qwen3NextAttention : public infinilm::weights::ModelWeights {
protected:
    infinilm::runtime::Device device_;

public:
    Qwen3NextAttention(infinilm::runtime::Runtime &runtime,
                   std::shared_ptr<infinilm::config::ModelConfig> model_config,
                       size_t layer_idx,
                   const engine::distributed::RankInfo &rank_info,
                   backends::AttentionBackend attention_backend,
                   const layers::moe::EPConfig &ep_config);

    infinilm::runtime::Tensor forward(const infinilm::runtime::Tensor &positions,
                               const infinilm::runtime::Tensor &hidden_states,
                               infinilm::global_state::ForwardContext &forward_context) const;
    void process_weights_after_loading(infinilm::runtime::Runtime &runtime, bool pre_transpose) override;

    size_t layer_idx() const { return layer_idx_; }
    size_t num_heads() const { return num_attention_heads_; }
    size_t num_kv_heads() const { return num_key_value_heads_; }
    size_t head_dim() const { return head_dim_; }
    size_t hidden_size() const { return hidden_size_; }

protected:
    infinilm::runtime::Tensor forward_static_(const infinilm::runtime::Tensor &position_ids,
                                       const infinilm::runtime::Tensor &hidden_states,
                                       infinilm::global_state::ForwardContext &forward_context) const;
    infinilm::runtime::Tensor forward_paged_(const infinilm::runtime::Tensor &position_ids,
                                      const infinilm::runtime::Tensor &hidden_states,
                                      infinilm::global_state::ForwardContext &forward_context) const;

    std::shared_ptr<infinilm::layers::linear::QKVParallelLinear> qkv_proj_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> o_proj_;
    std::shared_ptr<infinilm::layers::RMSNorm> q_norm_;
    std::shared_ptr<infinilm::layers::RMSNorm> k_norm_;
    std::shared_ptr<infinilm::layers::RoPE> rotary_emb_;

    std::shared_ptr<infinilm::layers::attention::AttentionLayer> attn_;
    ::infinilm::backends::AttentionBackend attention_backend_;
    size_t layer_idx_;
    size_t num_attention_heads_;
    size_t num_key_value_heads_;
    size_t hidden_size_;
    size_t head_dim_;

    // For off-line kv cache quantization
    infinilm::weights::Weight kv_cache_k_scale_;
    infinilm::weights::Weight kv_cache_v_scale_;
};

} // namespace infinilm::models::qwen3_next
