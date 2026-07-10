#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/attention/attention.hpp"
#include "../../layers/common_modules.hpp"
#include "../../layers/linear/linear.hpp"
#include "../../layers/rotary_embedding/rotary_embedding.hpp"
#include "infinicore/device.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <memory>

namespace infinilm::models::ernie4_5_vl {

class Ernie45Attention : public infinicore::nn::Module {
public:
    Ernie45Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                     size_t layer_idx,
                     const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               const infinicore::Tensor &hidden_states) const;

private:
    infinicore::Tensor forward_static_(const infinicore::Tensor &positions,
                                       const infinicore::Tensor &hidden_states) const;
    infinicore::Tensor forward_paged_(const infinicore::Tensor &positions,
                                      const infinicore::Tensor &hidden_states) const;

    size_t layer_idx_{0};
    size_t hidden_size_{0};
    size_t head_dim_{0};
    size_t rotary_dim_{0};
    size_t num_attention_heads_{0};
    size_t num_key_value_heads_{0};
    infinilm::backends::AttentionBackend attention_backend_;

    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, q_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, k_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, v_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::RowParallelLinear, o_proj);
    INFINICORE_NN_MODULE(infinicore::nn::RoPE, rotary_emb);

    infinicore::nn::Parameter kv_cache_k_scale_;
    infinicore::nn::Parameter kv_cache_v_scale_;
    std::shared_ptr<infinilm::layers::attention::AttentionLayer> attn_;
};

} // namespace infinilm::models::ernie4_5_vl
