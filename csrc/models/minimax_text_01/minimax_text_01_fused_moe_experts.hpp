#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/moe/common/moe_types.hpp"
#include "infinicore/nn/module.hpp"

#include <cstddef>
#include <memory>

namespace infinilm::models::minimax_text_01 {

// MXFP4 per-expert packed weights (w1/w3 share the w13 storage), see kimi_k3.
struct MiniMaxText01Mxfp4MoeWeights {
    infinicore::Tensor packed_w13; // [E, 2*I, H/2] U8
    infinicore::Tensor w13_scale;  // [E, 2*I, H/32] U8
    infinicore::Tensor packed_w2;  // [E, H, I/2] U8
    infinicore::Tensor w2_scale;   // [E, H, I/32] U8
};

/**
 * @brief MiniMax-Text-01 specific `FusedMoeExperts`.
 *
 * Copied from the shared `layers/moe/experts/fused_moe_experts`, with the
 * following differences:
 *  - the packed weights `w13_weight` / `w2_weight` are registered with an
 *    explicit `tp_dim` (w13 is split along its middle dimension, dim1, and w2
 *    along dim2) and a full shape (constructing an `nn::Parameter` with a
 *    `tp_dim` splits it automatically by `tp_size`), so that weights are
 *    partitioned per rank correctly when `TP > 1`;
 *  - when the config's `quantization_method` is MXFP4, it falls back to the
 *    per-expert packed registration (`N.w1/w2/w3.weight_packed` +
 *    `.weight_scale`, w1/w3 share the w13 storage), consumed by the
 *    `fused_moe_mxfp4` kernel for real 4-bit memory compression.
 *
 * The goal is to leave the shared `FusedMoeExperts` untouched (the other
 * models use it, so modifying shared code is forbidden) and only reproduce
 * the same correct behaviour inside the MiniMax-specific directory.
 */
class MiniMaxText01FusedMoeExperts : public infinicore::nn::Module {
public:
    MiniMaxText01FusedMoeExperts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                 const infinicore::Device &device);

    const infinilm::layers::moe::MoeWeights &moe_weights() const;
    bool use_mxfp4() const { return use_mxfp4_; }
    const MiniMaxText01Mxfp4MoeWeights &mxfp4_weights() const { return mxfp4_weights_; }

protected:
    void register_mxfp4_experts();

    INFINICORE_NN_PARAMETER(w13_weight);
    INFINICORE_NN_PARAMETER(w2_weight);

    size_t num_experts_{0};
    size_t hidden_size_{0};
    size_t intermediate_size_per_partition_{0};
    bool use_mxfp4_{false};
    infinicore::Device device_;
    infinilm::layers::moe::MoeWeights moe_weights_;
    MiniMaxText01Mxfp4MoeWeights mxfp4_weights_;
};

} // namespace infinilm::models::minimax_text_01
