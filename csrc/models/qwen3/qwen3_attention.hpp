#include "../../weights/weights.hpp"
#include "../../layers/normalization/rmsnorm.hpp"
#include "../../layers/rotary_embedding/rope.hpp"
#include "../models_registry.hpp"
#pragma once
#include <typeinfo>

#include "../../global_state/forward_context.hpp"
#include "../../layers/common_modules.hpp"

namespace infinilm::models::qwen3 {
class Qwen3Attention : public infinilm::weights::ModelWeights {
public:
    Qwen3Attention(infinilm::runtime::Runtime &, std::shared_ptr<config::ModelConfig>, size_t,
                   const engine::distributed::RankInfo &, backends::AttentionBackend, const layers::moe::EPConfig &);

    infinilm::runtime::Tensor forward(
        const infinilm::runtime::Tensor &positions,
        const infinilm::runtime::Tensor &hidden_states,
        infinilm::global_state::ForwardContext &forward_context) const;

    // Owner-local enumeration only; no new child/parameter registration.
    void append_w4_layers(std::vector<std::shared_ptr<infinilm::nn::BaseLinear>> &layers,
                          infinilm::quantization::QuantScheme expected_scheme) const {
        if (!qkv_proj_ || typeid(*qkv_proj_) != typeid(infinilm::layers::linear::QKVParallelLinear) ||
            qkv_proj_->get_quantization()->get_quant_scheme() != expected_scheme)
            throw std::invalid_argument("Qwen3 hidden QKV must be exact homogeneous W4 owner");
        layers.push_back(qkv_proj_);
    }

    void process_weights_after_loading(infinilm::runtime::Runtime &runtime, bool pre_transpose) override {
        qkv_proj_->process_weights_after_loading(runtime, pre_transpose);
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
} // namespace infinilm::models::qwen3
