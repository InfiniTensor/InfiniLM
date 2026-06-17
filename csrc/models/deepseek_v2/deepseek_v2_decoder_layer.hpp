#pragma once

#include "../../config/model_config.hpp"
#include "deepseek_v2_attention.hpp"
#include "deepseek_v2_mla_attention.hpp"
#include "deepseek_v2_moe.hpp"
#include "infinicore/device.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <tuple>
#include <variant>

namespace infinilm::models::deepseek_v2 {

using DeepseekV2SelfAttention = std::variant<std::shared_ptr<DeepseekV2Attention>, std::shared_ptr<DeepseekV2MLAAttention>>;

class DeepseekV2DecoderLayer : public infinicore::nn::Module {
public:
    DeepseekV2DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           size_t layer_idx,
                           const infinicore::Device &device);

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(const infinicore::Tensor &positions,
                                                               infinicore::Tensor &hidden_states,
                                                               infinicore::Tensor &residual) const;

private:
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(DeepseekV2SelfAttention, self_attn);
    INFINICORE_NN_MODULE(DeepseekV2MLP, dense_mlp);
    INFINICORE_NN_MODULE(DeepseekV2MoE, moe_mlp);
    bool use_moe_{false};
};

} // namespace infinilm::models::deepseek_v2
