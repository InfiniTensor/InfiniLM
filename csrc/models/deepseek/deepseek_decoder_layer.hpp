#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/attention/attention.hpp"
#include "../deepseek_v2/deepseek_v2_moe.hpp"
#include "infinicore/device.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <tuple>
#include <variant>

namespace infinilm::models::deepseek {

using DeepseekMLP = std::variant<std::shared_ptr<deepseek_v2::DeepseekV2MLP>,
                                 std::shared_ptr<deepseek_v2::DeepseekV2MoE>>;

using DeepseekAttention = infinilm::layers::attention::Attention;

class DeepseekDecoderLayer : public infinicore::nn::Module {
public:
    DeepseekDecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         size_t layer_idx,
                         const infinicore::Device &device);

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(const infinicore::Tensor &positions,
                                                               infinicore::Tensor &hidden_states,
                                                               infinicore::Tensor &residual);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               infinicore::Tensor &hidden_states);

private:
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(DeepseekAttention, self_attn);
    INFINICORE_NN_MODULE(DeepseekMLP, mlp);
};

} // namespace infinilm::models::deepseek
