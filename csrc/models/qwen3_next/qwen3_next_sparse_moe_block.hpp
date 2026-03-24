#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/common_modules.hpp"
#include "infinicore/device.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include "../../engine/distributed/distributed.hpp"

namespace infinilm::models::qwen3_next {

using Qwen3NextMLP = infinilm::layers::MoeMLP;

class Qwen3NextSparseMoeBlock : public infinicore::nn::Module {
public:
    Qwen3NextSparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                            const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

protected:
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, gate);
    INFINICORE_NN_MODULE_VEC(Qwen3NextMLP, experts);
    INFINICORE_NN_MODULE(Qwen3NextMLP, shared_expert);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, shared_expert_gate);
};

} // namespace infinilm::models::qwen3_next
