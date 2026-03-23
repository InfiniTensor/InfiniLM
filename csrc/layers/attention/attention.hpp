#pragma once

#include "../../backends/attention_backends.hpp"
#include "../../cache/kv_cache.hpp"
#include "../../config/infinilm_config.hpp"
#include "../../config/model_config.hpp"
#include "../../engine/distributed/distributed.hpp"
#include "../../models/infinilm_model.hpp"
#include "../../utils.hpp"
#include "../linear/fused_linear.hpp"
#include "backends/attention_layer.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/tensor.hpp"
#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <variant>

namespace infinilm::layers::attention {
class Attention : public infinicore::nn::Module {
public:
    Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
              size_t layer_idx,
              const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    infinicore::Tensor forward_paged_(const infinicore::Tensor &hidden_states,
                                      const infinilm::InfinilmModel::Input &attn_metadata,
                                      std::tuple<infinicore::Tensor, infinicore::Tensor>) const;

    infinicore::Tensor forward_static_(const infinicore::Tensor &hidden_states,
                                       const infinilm::InfinilmModel::Input &attn_metadata,
                                       std::tuple<infinicore::Tensor, infinicore::Tensor>) const;

public:
    size_t layer_idx() const { return layer_idx_; }
    size_t num_heads() const { return num_attention_heads_; }
    size_t num_kv_heads() const { return num_key_value_heads_; }
    size_t head_dim() const { return head_dim_; }
    size_t hidden_size() const { return hidden_size_; }
    void set_rotary_emb(const std::shared_ptr<infinicore::nn::RoPE> &rotary_emb) { rotary_emb_ = rotary_emb; }

protected:
    INFINICORE_NN_MODULE(infinilm::layers::linear::QKVParallelLinear, qkv_proj);
    INFINICORE_NN_MODULE(infinicore::nn::RowParallelLinear, o_proj);
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
    bool qk_norm_;
};
} // namespace infinilm::layers::attention