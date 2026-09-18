#pragma once

#include "../../layers/common_modules.hpp"

namespace infinilm::models::gemma2 {

class Gemma2MLP : public infinicore::nn::Module {
public:
    Gemma2MLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
              const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

protected:
    std::shared_ptr<infinilm::layers::linear::GateUpParallelLinear> gate_up_proj_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> down_proj_;
};

} // namespace infinilm::models::gemma2
