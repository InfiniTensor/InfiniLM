#pragma once

#include "../../backends/attention_backends.hpp"
#include "../../config/infinilm_config.hpp"
#include "../../config/model_config.hpp"
#include "../../engine/distributed/distributed.hpp"
#include "../../engine/forward_context.hpp"
#include "../../models/infinilm_model.hpp"
#include "../../utils.hpp"
#include "../linear/linear.hpp"
#include "backends/attention_layer.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/tensor.hpp"
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

namespace infinilm::layers::attention {
class Attention : public infinicore::nn::Module {
public:
    Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
              size_t layer_idx,
              const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

    size_t layer_idx() const { return layer_idx_; }
    size_t num_heads() const { return num_attention_heads_; }
    size_t num_kv_heads() const { return num_key_value_heads_; }
    size_t head_dim() const { return head_dim_; }
    size_t hidden_size() const { return hidden_size_; }
    void set_rotary_emb(const std::shared_ptr<infinicore::nn::RoPE> &rotary_emb) { rotary_emb_ = rotary_emb; }

private:
    infinicore::Tensor forward_static_(const infinicore::Tensor &hidden_states,
                                       const infinilm::engine::AttentionMetadata &attn_metadata,
                                       std::tuple<infinicore::Tensor, infinicore::Tensor> &kv_cache) const;

    infinicore::Tensor forward_paged_(const infinicore::Tensor &hidden_states,
                                      const infinilm::engine::AttentionMetadata &attn_metadata,
                                      std::tuple<infinicore::Tensor, infinicore::Tensor> &kv_cache) const;

protected:
    INFINICORE_NN_MODULE(infinilm::layers::linear::QKVParallelLinear, qkv_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::RowParallelLinear, o_proj);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, q_norm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, k_norm);

    std::shared_ptr<infinicore::nn::RoPE> rotary_emb_;
    std::shared_ptr<AttentionLayer> attn_;
    ::infinilm::backends::AttentionBackend attention_backend_;
    size_t layer_idx_;
    size_t num_attention_heads_;
    size_t num_key_value_heads_;
    size_t hidden_size_;
    size_t head_dim_;
};
} // namespace infinilm::layers::attention
