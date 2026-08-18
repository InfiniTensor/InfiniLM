#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/mlp/mlp.hpp"
#include "kimi_k25_attention.hpp"
#include "kimi_k25_moe.hpp"

#include <infinicore/nn/module.hpp>
#include <infinicore/nn/rmsnorm.hpp>
#include <infinicore/tensor.hpp>

#include <cstddef>
#include <memory>
#include <tuple>

namespace infinilm::models::kimi_k25 {

class KimiK25DecoderLayer : public infinicore::nn::Module {
public:
    KimiK25DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                        size_t layer_idx,
                        const infinicore::Device &device);

    std::tuple<infinicore::Tensor, infinicore::Tensor>
    forward(const infinicore::Tensor &positions,
            infinicore::Tensor &hidden_states,
            infinicore::Tensor &residual) const;

protected:
    INFINICORE_NN_MODULE(KimiK25Attention, self_attn);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(infinilm::layers::mlp::MLP, dense_mlp);
    INFINICORE_NN_MODULE(KimiK25MoE, moe_mlp);
    bool use_moe_{false};
};

} // namespace infinilm::models::kimi_k25
