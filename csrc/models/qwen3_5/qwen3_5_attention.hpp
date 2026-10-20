#pragma once
#include "../../layers/normalization/rmsnorm.hpp"
#include "../../layers/rotary_embedding/rope.hpp"
#include "../../weights/weights.hpp"

#include "../../layers/common_modules.hpp"
#include "qwen3_5_fused_qkv_linear.hpp"

namespace infinilm::models::qwen3_5 {
class Qwen35Attention : public infinilm::weights::ModelWeights {
protected:
    infinilm::runtime::Device device_;

public:
    Qwen35Attention(infinilm::runtime::Runtime &runtime,
                    std::shared_ptr<infinilm::config::ModelConfig> model_config,
                    size_t layer_idx, const engine::distributed::RankInfo &rank_info,
                    ::infinilm::backends::AttentionBackend attention_backend);

    infinilm::runtime::Tensor forward(const infinilm::runtime::Tensor &positions,
                                      const infinilm::runtime::Tensor &hidden_states,
                                      infinilm::global_state::ForwardContext &forward_context) const;

    using infinilm::weights::ModelWeights::process_weights_after_loading;
    std::vector<std::shared_ptr<infinilm::weights::ModelWeights>> loading_children() const override {
        auto result = ModelWeights::loading_children();
        if (qkv_proj_) {
            result.push_back(qkv_proj_);
        }
        return result;
    }

    size_t layer_idx() const { return layer_idx_; }
    size_t num_heads() const { return num_attention_heads_; }
    size_t num_kv_heads() const { return num_key_value_heads_; }
    size_t head_dim() const { return head_dim_; }
    size_t hidden_size() const { return hidden_size_; }

private:
    infinilm::runtime::Tensor forward_static_(const infinilm::runtime::Tensor &positions,
                                              const infinilm::runtime::Tensor &hidden_states,
                                              infinilm::global_state::ForwardContext &forward_context) const;

    infinilm::runtime::Tensor forward_paged_(const infinilm::runtime::Tensor &positions,
                                             const infinilm::runtime::Tensor &hidden_states,
                                             infinilm::global_state::ForwardContext &forward_context) const;

protected:
    std::shared_ptr<Qwen35FusedQKVLinear> qkv_proj_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> o_proj_;
    std::shared_ptr<infinilm::layers::RMSNorm> q_norm_;
    std::shared_ptr<infinilm::layers::RMSNorm> k_norm_;
    std::shared_ptr<infinilm::layers::RoPE> mrope_;

    std::shared_ptr<infinilm::layers::attention::AttentionLayer> attn_;
    ::infinilm::backends::AttentionBackend attention_backend_;
    size_t layer_idx_;
    size_t num_attention_heads_;
    size_t num_key_value_heads_;
    size_t hidden_size_;
    size_t head_dim_;
    size_t rotary_dim_;
    infinilm::runtime::Runtime *construction_owner_ = nullptr; // Borrowed worker lifetime, not a new Runtime/cache.

    infinilm::weights::Weight kv_cache_k_scale_;
    infinilm::weights::Weight kv_cache_v_scale_;
};
} // namespace infinilm::models::qwen3_5
