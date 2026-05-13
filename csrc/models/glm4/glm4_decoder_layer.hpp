#pragma once

#include "../../config/model_config.hpp"
#include "../../backends/attention_backends.hpp"
#include "../../engine/distributed/distributed.hpp"
#include "glm4_attention.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "../../layers/mlp/mlp.hpp"

namespace infinilm::models::glm4 {

class Glm4DecoderLayer : public infinicore::nn::Module {
public:
    Glm4DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                     size_t layer_idx,
                     const infinicore::Device &device);

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(
        const infinicore::Tensor &positions,
        infinicore::Tensor &hidden_states,
        infinicore::Tensor &residual);

    infinicore::Tensor forward(
        const infinicore::Tensor &positions,
        infinicore::Tensor &hidden_states);

private:
    INFINICORE_NN_MODULE(Glm4Attention, self_attn);
    INFINICORE_NN_MODULE(infinilm::layers::mlp::MLP, mlp);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_self_attn_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_mlp_layernorm);
};

} // namespace infinilm::models::glm4

