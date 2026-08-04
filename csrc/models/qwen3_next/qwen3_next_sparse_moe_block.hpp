#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/linear/linear.hpp"
#include "../../layers/mlp/mlp.hpp"
#include "../../layers/moe/sparse_moe_block.hpp"
#include "infinicore/nn/module.hpp"

#include <cstddef>
#include <memory>

namespace infinilm::models::qwen3_next {

class Qwen3NextSparseMoeBlock : public infinilm::layers::moe::SparseMoeBlock {
public:
    Qwen3NextSparseMoeBlock(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const infinicore::Device &device);
    Qwen3NextSparseMoeBlock(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        size_t layer_idx,
        const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

protected:
    INFINICORE_NN_MODULE(infinilm::layers::mlp::MLP, shared_expert);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, shared_expert_gate);
};

} // namespace infinilm::models::qwen3_next
