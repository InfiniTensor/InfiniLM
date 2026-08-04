#pragma once

#include "kimi_k3_delta_attention.hpp"
#include "kimi_k3_mla_attention.hpp"
#include "kimi_k3_moe.hpp"

#include <infinicore/nn/module.hpp>
#include <infinicore/nn/rmsnorm.hpp>
#include <infinicore/tensor.hpp>

#include <memory>
#include <utility>

namespace infinilm::models::kimi_k3 {

class KimiK3DecoderLayer : public infinicore::nn::Module {
public:
    KimiK3DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                       size_t layer_idx,
                       const infinicore::Device &device);

    std::pair<infinicore::Tensor, infinicore::Tensor>
    forward(const infinicore::Tensor &hidden_states,
            const infinicore::Tensor &block_residual) const;

private:
    infinicore::Tensor apply_attn_res(const infinicore::Tensor &prefix_sum,
                                      const infinicore::Tensor &block_residual,
                                      const std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> &proj,
                                      const std::shared_ptr<infinicore::nn::RMSNorm> &norm) const;

    size_t layer_idx_{0};
    size_t hidden_size_{0};
    size_t attn_res_block_size_{12};
    bool is_kda_{false};
    bool use_moe_{false};

    INFINICORE_NN_MODULE(KimiK3DeltaAttention, delta_attn);
    INFINICORE_NN_MODULE(KimiK3MLAAttention, mla_attn);
    INFINICORE_NN_MODULE(KimiK3MoE, block_sparse_moe);
    INFINICORE_NN_MODULE(KimiK3MLP, mlp);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, self_attention_res_norm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, mlp_res_norm);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, self_attention_res_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, mlp_res_proj);
};

} // namespace infinilm::models::kimi_k3
