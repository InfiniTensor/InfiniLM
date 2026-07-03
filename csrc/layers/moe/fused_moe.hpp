#pragma once

#include "common/moe_types.hpp"
#include "dispatcher/base_dispatcher.hpp"
#include "runner/base_runner.hpp"

#include "../../config/model_config.hpp"
#include "infinicore/nn/module.hpp"

#include <memory>

namespace infinilm::layers::moe {

class FusedMoE final : public infinicore::nn::Module {
public:
    FusedMoE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
             const infinicore::Device &device,
             size_t layer_id = 0);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const TopKOutput &topk_output,
                               const MoeWeights &weights) const;

private:
    std::shared_ptr<BaseDispatcher> dispatcher_;
    std::shared_ptr<MoeRunnerCore> runner_;
    mutable MoeWorkspace workspace_;
};

} // namespace infinilm::layers::moe
