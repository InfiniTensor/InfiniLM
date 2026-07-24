#pragma once

#include "../../layers/attention/attention.hpp"
#include "../../layers/linear/fused_linear.hpp"
#include "../../layers/rotary_embedding/rotary_embedding.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <memory>

namespace infinilm::models::ernie4_5_moe_vl {

class Ernie4_5Attention : public infinicore::nn::Module {
public:
    Ernie4_5Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                      size_t layer_idx,
                      const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               const infinicore::Tensor &hidden_states) const;

private:
    infinicore::Tensor forward_static_(const infinicore::Tensor &positions,
                                       const infinicore::Tensor &hidden_states) const;
    infinicore::Tensor forward_paged_(const infinicore::Tensor &positions,
                                      const infinicore::Tensor &hidden_states) const;
    infinicore::Tensor position_ids_for_rope_(const infinicore::Tensor &positions) const;
    bool should_use_rope_3d_(const infinicore::Tensor &positions) const;
    infinicore::Tensor apply_rope_3d_(const infinicore::Tensor &states,
                                      const infinicore::Tensor &positions) const;

    std::shared_ptr<infinilm::layers::linear::QKVParallelLinear> qkv_proj_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> o_proj_;
    std::shared_ptr<infinicore::nn::RoPE> rotary_emb_;
    std::shared_ptr<infinilm::layers::attention::AttentionLayer> attn_;
    infinilm::backends::AttentionBackend attention_backend_;

    size_t layer_idx_{0};
    size_t num_attention_heads_{0};
    size_t num_key_value_heads_{0};
    size_t hidden_size_{0};
    size_t head_dim_{0};
    size_t freq_allocation_{20};
    double rope_theta_{10000.0};
    double compression_ratio_{1.0};
    bool use_rope_3d_{false};

    INFINICORE_NN_PARAMETER(kv_cache_k_scale);
    INFINICORE_NN_PARAMETER(kv_cache_v_scale);
};

} // namespace infinilm::models::ernie4_5_moe_vl
