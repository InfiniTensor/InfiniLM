
#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/common_modules.hpp"
#include "infinicore/device.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include "../../engine/distributed/distributed.hpp"

namespace infinilm::models::qwen3_next {
using Qwen3Next_Fake_RMSNormGated = infinicore::nn::RMSNorm;

class FakeConv1d : public infinicore::nn::Module {
public:
    FakeConv1d(size_t in_channels,
           size_t out_channels,
           size_t kernel_size,
           size_t stride,
           size_t padding,
           size_t dilation,
           size_t groups,
           bool bias,
           const infinicore::DataType dtype,
           const infinicore::Device device ) {

        INFINICORE_NN_PARAMETER_INIT(weight, ({out_channels,1,kernel_size}, dtype, device));
    }

private:
    size_t layer_idx_;
    INFINICORE_NN_PARAMETER(weight);
};

class Qwen3NextGatedDeltaNet : public infinicore::nn::Module {
public:
    Qwen3NextGatedDeltaNet(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           size_t layer_idx,
                           const infinicore::Device &device,
                           engine::distributed::RankInfo rank_info) {
        const auto &dtype{model_config->get_dtype()};
        layer_idx_ = layer_idx;
        size_t hidden_size = (model_config->get<size_t>("hidden_size"));
        size_t linear_num_value_heads = (model_config->get<size_t>("linear_num_value_heads"));
        size_t linear_num_key_heads = (model_config->get<size_t>("linear_num_key_heads"));
        size_t linear_key_head_dim = (model_config->get<size_t>("linear_key_head_dim"));
        size_t linear_value_head_dim = (model_config->get<size_t>("linear_value_head_dim"));

        size_t key_dim = linear_key_head_dim * linear_num_key_heads;
        size_t value_dim = linear_value_head_dim * linear_num_value_heads;

        size_t linear_conv_kernel_dim = model_config->get<size_t>("linear_conv_kernel_dim");

        double rms_norm_eps = model_config->get<double>("rms_norm_eps");
        // QKV
        size_t conv_dim = key_dim * 2 + value_dim;
        INFINICORE_NN_MODULE_INIT(conv1d, conv_dim, conv_dim, linear_conv_kernel_dim, 1, linear_conv_kernel_dim - 1, 1, 1, false, dtype, device);

        size_t projection_size_qkvz = key_dim * 2 + value_dim * 2;
        size_t projection_size_ba = linear_num_value_heads * 2;

        INFINICORE_NN_MODULE_INIT(in_proj_qkvz, hidden_size, projection_size_qkvz, false, dtype, device);
        INFINICORE_NN_MODULE_INIT(in_proj_ba, hidden_size, projection_size_ba, false, dtype, device);

        // # time step projection (discretization)
        // # instantiate once and copy inv_dt in init_weights of PretrainedModel
        INFINICORE_NN_PARAMETER_INIT(dt_bias, ({linear_num_value_heads}, dtype, device));
        INFINICORE_NN_PARAMETER_INIT(A_log, ({linear_num_value_heads}, dtype, device));

        INFINICORE_NN_MODULE_INIT(norm, linear_value_head_dim, rms_norm_eps, dtype, device);
        INFINICORE_NN_MODULE_INIT(out_proj,value_dim, hidden_size,  false, dtype, device);
    }

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const {
        throw std::runtime_error("Qwen3NextGatedDeltaNet: forward not implemented");
        return hidden_states;
    }

private:
    size_t layer_idx_;

    INFINICORE_NN_MODULE(infinicore::nn::Linear, in_proj_qkvz);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, in_proj_ba);
    INFINICORE_NN_MODULE(FakeConv1d, conv1d);
    INFINICORE_NN_PARAMETER(dt_bias);
    INFINICORE_NN_PARAMETER(A_log);
    INFINICORE_NN_MODULE(Qwen3Next_Fake_RMSNormGated, norm);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, out_proj);
};

} // namespace infinilm::models::qwen3_next