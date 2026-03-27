#pragma once

#include "../../layers/common_modules.hpp"

namespace infinilm::models::qwen3_moe {
using Qwen3MoeMLP = infinilm::layers::MoeMLP;

class Qwen3MoeSparseMoeBlock : public infinicore::nn::Module {
public:
    Qwen3MoeSparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

protected:
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, gate);
    INFINICORE_NN_MODULE_VEC(Qwen3MoeMLP, experts);
    INFINICORE_NN_MODULE(Qwen3MoeMLP, shared_expert);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, shared_expert_gate);
};

} // namespace infinilm::models::qwen3_moe
