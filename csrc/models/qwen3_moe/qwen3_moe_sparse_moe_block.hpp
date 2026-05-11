#pragma once

#include "qwen3_moe_experts.hpp"
#include "qwen3_moe_topk_router.hpp"

namespace infinilm::models::qwen3_moe {

class Qwen3MoeSparseMoeBlock : public infinicore::nn::Module {
public:
    Qwen3MoeSparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

protected:
    INFINICORE_NN_MODULE(Qwen3MoeTopKRouter, gate);
    INFINICORE_NN_MODULE(Qwen3MoeExperts, experts);
};

} // namespace infinilm::models::qwen3_moe
