#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/mlp/mlp.hpp"
#include "minicpm5_moe_attention.hpp"
#include "minicpm5_moe_sparse_moe_block.hpp"

#include <infinicore/nn/module.hpp>
#include <infinicore/nn/rmsnorm.hpp>
#include <infinicore/tensor.hpp>

#include <cstddef>
#include <memory>
#include <tuple>

namespace infinilm::models::minicpm5_moe {

class MiniCPM5MoEDecoderLayer : public infinicore::nn::Module {
public:
    MiniCPM5MoEDecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                            size_t layer_idx,
                            const infinicore::Device &device);

    std::tuple<infinicore::Tensor, infinicore::Tensor>
    forward(const infinicore::Tensor &positions,
            infinicore::Tensor &hidden_states,
            infinicore::Tensor &residual) const;

protected:
    INFINICORE_NN_MODULE(MiniCPM5MoEAttention, self_attn);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(infinilm::layers::mlp::MLP, dense_mlp);
    INFINICORE_NN_MODULE(MiniCPM5MoESparseMoeBlock, moe_mlp);
    bool use_moe_{false};
};

} // namespace infinilm::models::minicpm5_moe
