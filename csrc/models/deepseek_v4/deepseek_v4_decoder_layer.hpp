#pragma once

#include "../../config/model_config.hpp"
#include "deepseek_v4_attention.hpp"
#include "deepseek_v4_moe.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/tensor.hpp"

#include <cstddef>
#include <memory>
#include <tuple>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4DecoderLayer : public infinicore::nn::Module {
public:
    DeepseekV4DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           const infinicore::Device &device);
    DeepseekV4DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           size_t layer_idx,
                           const infinicore::Device &device);

    std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor, infinicore::Tensor>
    forward(const infinicore::Tensor &hidden_states,
            const infinicore::Tensor &positions,
            const infinicore::Tensor &input_ids = infinicore::Tensor(),
            const infinicore::Tensor &post_mix = infinicore::Tensor(),
            const infinicore::Tensor &res_mix = infinicore::Tensor(),
            const infinicore::Tensor &residual = infinicore::Tensor()) const;

private:
    INFINICORE_NN_MODULE(DeepseekV4Attention, attn);
    INFINICORE_NN_MODULE(DeepseekV4MoE, ffn);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, attn_norm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, ffn_norm);
    INFINICORE_NN_PARAMETER(hc_attn_base);
    INFINICORE_NN_PARAMETER(hc_attn_fn);
    INFINICORE_NN_PARAMETER(hc_attn_scale);
    INFINICORE_NN_PARAMETER(hc_ffn_base);
    INFINICORE_NN_PARAMETER(hc_ffn_fn);
    INFINICORE_NN_PARAMETER(hc_ffn_scale);

    size_t layer_idx_{0};
    size_t hidden_size_{0};
    size_t hc_mult_{0};
    size_t hc_sinkhorn_iters_{0};
    double hc_eps_{0.0};
    double hc_post_alpha_{2.0};
};

} // namespace infinilm::models::deepseek_v4
