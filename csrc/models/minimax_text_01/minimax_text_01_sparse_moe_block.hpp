#pragma once

#include "../../layers/moe/fused_moe.hpp"
#include "../../layers/moe/router/topk_router.hpp"
#include "minimax_text_01_fused_moe_experts.hpp"

#include "../../config/model_config.hpp"
#include "infinicore/nn/module.hpp"

#include <cstddef>
#include <memory>

namespace infinilm::models::minimax_text_01 {

/**
 * @brief MiniMax-Text-01 specific MoE block.
 *
 * Copied from the shared `layers/moe/sparse_moe_block`; only the experts are
 * replaced with `MiniMaxText01FusedMoeExperts` (which splits w13/w2 correctly
 * under TP). The gate (`TopKRouter`) and the fused MoE (`FusedMoE`) reuse the
 * shared components unchanged.
 */
class MiniMaxText01SparseMoeBlock : public infinicore::nn::Module {
public:
    MiniMaxText01SparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                const infinicore::Device &device,
                                size_t layer_id = 0);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

protected:
    INFINICORE_NN_MODULE(infinilm::layers::moe::TopKRouter, gate);
    INFINICORE_NN_MODULE(MiniMaxText01FusedMoeExperts, experts);
    INFINICORE_NN_MODULE(infinilm::layers::moe::FusedMoE, fused_moe);
};

} // namespace infinilm::models::minimax_text_01
