#pragma once

#include "../../layers/moe/legacy/moe_mlp.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <cstddef>
#include <memory>

namespace infinilm::models::qwen3_moe {

using Qwen3MoeMLP = infinilm::layers::moe::legacy::MoeMLP;

class Qwen3MoeExperts : public infinicore::nn::Module {
public:
    Qwen3MoeExperts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                    const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &top_k_index,
                               const infinicore::Tensor &top_k_weights) const;

protected:
    INFINICORE_NN_MODULE_VEC(Qwen3MoeMLP, experts);
    size_t num_experts_per_tok_{0};
    size_t num_experts_{0};
};

} // namespace infinilm::models::qwen3_moe
