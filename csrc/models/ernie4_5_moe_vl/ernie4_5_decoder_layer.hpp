#pragma once

#include "../../layers/mlp/mlp.hpp"
#include "ernie4_5_attention.hpp"
#include "ernie4_5_moe.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/tensor.hpp"

#include <memory>

namespace infinilm::models::ernie4_5_moe_vl {

class Ernie4_5DecoderLayer : public infinicore::nn::Module {
public:
    Ernie4_5DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         size_t layer_idx,
                         const infinicore::Device &device);

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(const infinicore::Tensor &positions,
                                                               infinicore::Tensor &hidden_states,
                                                               infinicore::Tensor &residual,
                                                               const infinicore::Tensor &token_type_ids = infinicore::Tensor());

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &token_type_ids = infinicore::Tensor());

private:
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(Ernie4_5Attention, self_attn);

    std::shared_ptr<infinilm::layers::mlp::MLP> dense_mlp_;
    std::shared_ptr<Ernie4_5TextMoeBlock> moe_mlp_;
    size_t layer_idx_{0};
};

} // namespace infinilm::models::ernie4_5_moe_vl
