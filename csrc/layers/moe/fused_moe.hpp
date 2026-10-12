#include "../../models/models_registry.hpp"
#pragma once

#include "common/moe_types.hpp"
#include "dispatcher/base_dispatcher.hpp"
#include "runner/base_runner.hpp"

#include "../../config/model_config.hpp"
#include "../../weights/weights.hpp"

#include <memory>

namespace infinilm::layers::moe {

class FusedMoE final : public infinilm::weights::ModelWeights {
public:
    FusedMoE(infinilm::runtime::Runtime &, std::shared_ptr<config::ModelConfig>,
        const engine::distributed::RankInfo &, const EPConfig &, size_t layer_id = 0);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                               const infinilm::runtime::Tensor &hidden_states,
                               const TopKOutput &topk_output,
                               const MoeWeights &weights) const;

private:
    infinilm::runtime::Runtime *runtime_borrowed_{nullptr}; // Borrowed; dispatcher retains legitimate shared owner.
    std::shared_ptr<BaseDispatcher> dispatcher_;
    std::shared_ptr<MoeRunnerCore> runner_;
    mutable MoeWorkspace workspace_;
};

} // namespace infinilm::layers::moe
