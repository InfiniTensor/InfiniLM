#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/linear/linear.hpp"
#include "infinicore/nn/layer_norm.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/tensor.hpp"

#include <memory>

namespace infinilm::models::ernie4_5_moe_vl {

class Ernie4_5Resampler : public infinicore::nn::Module {
public:
    Ernie4_5Resampler(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                      const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &vision_features,
                               const infinicore::Tensor &grid_thw) const;

private:
    infinicore::Tensor temporal_placeholder_(const infinicore::Tensor &x,
                                             const infinicore::Tensor &grid_thw) const;

    size_t in_dim_{0};
    size_t out_dim_{0};
    size_t spatial_conv_size_{2};
    size_t temporal_conv_size_{2};
    size_t spatial_dim_{0};
    size_t temporal_dim_{0};
    bool use_temporal_conv_{true};

    INFINICORE_NN_MODULE(infinilm::nn::Linear, spatial_linear_0);
    INFINICORE_NN_MODULE(infinilm::nn::Linear, spatial_linear_2);
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, spatial_linear_3);
    INFINICORE_NN_MODULE(infinilm::nn::Linear, temporal_linear_0);
    INFINICORE_NN_MODULE(infinilm::nn::Linear, temporal_linear_2);
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, temporal_linear_3);
    INFINICORE_NN_MODULE(infinilm::nn::Linear, mlp);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, after_norm);
};

} // namespace infinilm::models::ernie4_5_moe_vl
