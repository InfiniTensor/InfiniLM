#pragma once

#include "../../layers/common_modules.hpp"
#include "../../layers/moe/experts/fused_moe_experts.hpp"
#include "../../layers/moe/fused_moe.hpp"
#include "../../layers/moe/router/topk_router.hpp"

#include <memory>

namespace infinilm::models::minimax {

// Block-sparse MoE for MiniMax (matching HF transformers `minimax`):
// softmax top-k router + gate_up_proj/down_proj experts. No shared MLP in the
// transformers `minimax` reference (shared_mlp + coefficient is a follow-up for
// MiniMax-M2 style configs).
class MiniMaxMoeBlock : public infinicore::nn::Module {
public:
    MiniMaxMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                    size_t layer_idx,
                    const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

protected:
    std::shared_ptr<infinilm::layers::moe::TopKRouter> gate_;
    std::shared_ptr<infinilm::layers::moe::FusedMoeExperts> experts_;
    std::shared_ptr<infinilm::layers::moe::FusedMoE> fused_moe_;
};

} // namespace infinilm::models::minimax
