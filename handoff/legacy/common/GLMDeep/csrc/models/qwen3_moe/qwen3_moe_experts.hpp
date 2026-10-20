#include "../models_registry.hpp"
#pragma once

#include "../../layers/moe/common/moe_types.hpp"

#include "../../weights/weights.hpp"
#include "../../weights/weights.hpp"
#include "runtime/tensor.hpp"

#include <cstddef>
#include <memory>

namespace infinilm::config {
class ModelConfig;
}

namespace infinilm::models::qwen3_moe {

class Qwen3MoeExperts : public infinilm::weights::ModelWeights {
public:
    Qwen3MoeExperts(infinilm::runtime::Runtime &, std::shared_ptr<config::ModelConfig>,
        const engine::distributed::RankInfo &, const layers::moe::EPConfig &);

    infinilm::layers::moe::MoeWeights moe_weights(infinilm::runtime::Runtime &runtime) const;

protected:
    infinilm::weights::Weight w1_;
    infinilm::weights::Weight w2_;

    size_t num_experts_per_tok_{0};
    size_t num_experts_{0};
    size_t hidden_size_{0};
    size_t intermediate_size_per_partition_{0};
    int tp_size_{1};
};

} // namespace infinilm::models::qwen3_moe
