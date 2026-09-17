#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/linear/linear.hpp"
#include "infinicore/nn/rmsnorm.hpp"

namespace infinilm::models::mamba2 {

infinicore::Tensor cast_activation(const infinicore::Tensor &input, infinicore::DataType dtype);

class Mamba2Mixer : public infinicore::nn::Module {
public:
    Mamba2Mixer(std::shared_ptr<config::ModelConfig> config, size_t layer_idx,
                const infinicore::Device &device);
    infinicore::Tensor forward(infinicore::Tensor input) const;
    void process_weights_after_loading() override { in_proj_->process_weights_after_loading(); }

private:
    size_t layer_idx_, intermediate_, heads_, head_dim_, state_size_, conv_dim_;
    size_t tp_rank_, tp_size_;
    infinicclComm_t communicator_;
    std::shared_ptr<layers::linear::QKVParallelLinear> in_proj_;
    INFINICORE_NN_MODULE(layers::linear::RowParallelLinear, out_proj);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);
    infinicore::Tensor conv1d_weight_, conv1d_bias_, norm_scale_, norm_epsilon_;
    INFINICORE_NN_PARAMETER(A);
    INFINICORE_NN_PARAMETER(D);
    INFINICORE_NN_PARAMETER(dt_bias);
};

class Mamba2Block : public infinicore::nn::Module {
public:
    Mamba2Block(std::shared_ptr<config::ModelConfig> config, size_t layer_idx,
                const infinicore::Device &device);
    infinicore::Tensor forward(const infinicore::Tensor &residual) const;

private:
    infinicore::DataType dtype_;
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);
    INFINICORE_NN_MODULE(Mamba2Mixer, mixer);
};

} // namespace infinilm::models::mamba2
