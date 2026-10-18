#pragma once

#include "../../backends/attention_backends.hpp"
#include "../../config/model_config.hpp"
#include "../../global_state/global_state.hpp"
#include "../linear/linear.hpp"
#include "backends/attention_layer.hpp"
#include "../../weights/weights.hpp"
#include "../rotary_embedding/rope.hpp"
#include "runtime/tensor.hpp"
#include <memory>

namespace infinilm::layers::moe { struct EPConfig; }

namespace infinilm::layers::attention {
class Attention : public infinilm::weights::ModelWeights {
public:
    std::vector<std::shared_ptr<infinilm::weights::ModelWeights>> loading_children() const override {
        auto result = ModelWeights::loading_children();
        if (qkv_proj_) result.push_back(qkv_proj_);
        return result;
    }

    Attention(infinilm::runtime::Runtime &runtime,
              std::shared_ptr<infinilm::config::ModelConfig> model_config,
              size_t layer_idx, const engine::distributed::RankInfo &rank_info,
              ::infinilm::backends::AttentionBackend attention_backend,
              const infinilm::layers::moe::EPConfig &ep_config);

    infinilm::runtime::Tensor forward(
        const infinilm::runtime::Tensor &positions,
        const infinilm::runtime::Tensor &hidden_states,
        infinilm::global_state::ForwardContext &forward_context) const;



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
    std::shared_ptr<infinilm::layers::RoPE> rotary_emb_;

    std::shared_ptr<AttentionLayer> attn_;
    ::infinilm::backends::AttentionBackend attention_backend_;
    size_t layer_idx_;
    size_t num_attention_heads_;
    size_t num_key_value_heads_;
    size_t hidden_size_;
    size_t head_dim_;
    infinilm::runtime::Runtime *construction_owner_ = nullptr; // Non-owning, worker-owned selected lifetime.

    // For off-line kv cache quantization
    infinilm::weights::Weight kv_cache_k_scale_;
    infinilm::weights::Weight kv_cache_v_scale_;
};
void init_kv_cache_quant_params(infinilm::runtime::Runtime &runtime,
                                std::function<void(const std::string &, infinilm::weights::Weight)> register_fn,
                                infinilm::weights::Weight &kv_cache_k_scale,
                                infinilm::weights::Weight &kv_cache_v_scale,
                                infinilm::quantization::KVQuantAlgo kv_quant_scheme);
namespace detail {
void apply_output_gate(infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor &output,
                       const infinilm::runtime::Tensor &gate, bool view_sigmoid_output);
}
} // namespace infinilm::layers::attention
