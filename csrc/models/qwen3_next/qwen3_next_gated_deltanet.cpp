#include "qwen3_next_gated_deltanet.hpp"
#include <spdlog/spdlog.h>

namespace infinilm::models::qwen3_next {

FakeConv1d::FakeConv1d(size_t in_channels,
                       size_t out_channels,
                       size_t kernel_size,
                       size_t stride,
                       size_t padding,
                       size_t dilation,
                       size_t groups,
                       bool bias,
                       const infinicore::DataType dtype,
                       const infinicore::Device device) {

    INFINICORE_NN_PARAMETER_INIT(weight, ({out_channels, 1, kernel_size}, dtype, device));
}

Qwen3NextGatedDeltaNet::Qwen3NextGatedDeltaNet(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                               size_t layer_idx,
                                               const infinicore::Device &device) {
    layer_idx_ = layer_idx;
    const auto &dtype{model_config->get_dtype()};
    size_t hidden_size = model_config->get<size_t>("hidden_size");
    size_t linear_num_value_heads = model_config->get<size_t>("linear_num_value_heads");
    size_t linear_num_key_heads = model_config->get<size_t>("linear_num_key_heads");
    size_t linear_key_head_dim = model_config->get<size_t>("linear_key_head_dim");
    size_t linear_value_head_dim = model_config->get<size_t>("linear_value_head_dim");

    size_t key_dim = linear_key_head_dim * linear_num_key_heads;
    size_t value_dim = linear_value_head_dim * linear_num_value_heads;

    size_t linear_conv_kernel_dim = model_config->get<size_t>("linear_conv_kernel_dim");

    double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    size_t conv_dim = key_dim * 2 + value_dim;
    INFINICORE_NN_MODULE_INIT(conv1d, conv_dim, conv_dim, linear_conv_kernel_dim, 1, linear_conv_kernel_dim - 1, 1, 1, false, dtype, device);

    size_t projection_size_qkvz = key_dim * 2 + value_dim * 2;
    size_t projection_size_ba = linear_num_value_heads * 2;

    INFINICORE_NN_MODULE_INIT(in_proj_qkvz, hidden_size, projection_size_qkvz, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(in_proj_ba, hidden_size, projection_size_ba, false, dtype, device);

    INFINICORE_NN_PARAMETER_INIT(dt_bias, ({linear_num_value_heads}, dtype, device));
    INFINICORE_NN_PARAMETER_INIT(A_log, ({linear_num_value_heads}, dtype, device));

    INFINICORE_NN_MODULE_INIT(norm, linear_value_head_dim, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(out_proj, value_dim, hidden_size, false, dtype, device);
}

infinicore::Tensor Qwen3NextGatedDeltaNet::forward(const infinicore::Tensor &hidden_states) const {
    spdlog::error("Qwen3NextGatedDeltaNet: forward not implemented");
    return hidden_states;
}

} // namespace infinilm::models::qwen3_next
