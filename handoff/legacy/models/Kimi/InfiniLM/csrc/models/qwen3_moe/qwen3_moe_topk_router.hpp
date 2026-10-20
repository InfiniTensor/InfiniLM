#include "../models_registry.hpp"
#pragma once

#include "../../config/model_config.hpp"
#include "runtime/device.hpp"
#include "../../weights/weights.hpp"
#include "runtime/tensor.hpp"

#include <memory>
#include <tuple>

namespace infinilm::models::qwen3_moe {

class Qwen3MoeTopKRouter : public infinilm::weights::ModelWeights {
public:
    Qwen3MoeTopKRouter(infinilm::runtime::Runtime &, std::shared_ptr<config::ModelConfig>,
        const engine::distributed::RankInfo &, const layers::moe::EPConfig &);

    std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor> forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const;
    void validate_runtime(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const;

protected:
    infinilm::weights::Weight weight_;

    infinilm::runtime::Runtime *runtime_borrowed_{nullptr}; // explicit constructor borrows retained caller owner.
    size_t num_experts_per_tok_{0};
    bool norm_topk_prob_{false};
};

} // namespace infinilm::models::qwen3_moe
