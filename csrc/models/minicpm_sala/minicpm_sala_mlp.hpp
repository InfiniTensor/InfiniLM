#pragma once

#include "../../config/model_config.hpp"

#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <memory>

namespace infinilm::models::minicpm_sala {

class MiniCPMSALAMLP : public infinicore::nn::Module {
public:
    MiniCPMSALAMLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                   const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &x) const;

protected:
    INFINICORE_NN_MODULE(infinicore::nn::Linear, gate_proj);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, up_proj);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, down_proj);
};

} // namespace infinilm::models::minicpm_sala

