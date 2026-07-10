#pragma once

#include "infinicore/nn/module.hpp"
#include "infinicore/nn/parameter.hpp"
#include "infinicore/tensor.hpp"

#include <cstddef>
#include <memory>

namespace infinilm::config {
class ModelConfig;
}

namespace infinilm::models::qwen3_moe {

class Qwen3MoeExperts : public infinicore::nn::Module {
public:
    Qwen3MoeExperts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                    const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &top_k_index,
                               const infinicore::Tensor &top_k_weights) const;

protected:
    INFINICORE_NN_PARAMETER(w1);
    INFINICORE_NN_PARAMETER(w2);

    size_t num_experts_per_tok_{0};
    size_t num_experts_{0};
    size_t hidden_size_{0};
    size_t intermediate_size_per_partition_{0};
};

} // namespace infinilm::models::qwen3_moe
