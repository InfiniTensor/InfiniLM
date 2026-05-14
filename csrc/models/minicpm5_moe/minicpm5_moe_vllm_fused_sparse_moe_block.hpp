#pragma once

#include "minicpm5_moe_sparse_moe_block.hpp"

#include <optional>

namespace infinilm::models::minicpm5_moe {

/**
 * MiniCPM5 MoE block: grouped sigmoid routing on device (``torch.ops.infinilm.minicpm5_grouped_sigmoid_topk``)
 * when the fused stack is available, else HF-aligned CPU routing; then vLLM ``fused_experts`` for experts.
 * Set ``INFINILM_MOE_FUSED_STACK=vendor_router_cpu`` to force the CPU router + pack path (same vendor fused experts as ``vendor``).
 * Falls back to
 * ``MiniCPM5MoeSparseMoeBlock::forward`` if dispatch fails or ``INFINILM_DISABLE_VLLM_FUSED_MOE=1``.
 */
class MiniCPM5MoeVllmFusedSparseMoeBlock : public MiniCPM5MoeSparseMoeBlock {
public:
    using MiniCPM5MoeSparseMoeBlock::MiniCPM5MoeSparseMoeBlock;

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const override;

private:
    void rebuild_stacked_expert_weights(const infinicore::Device &dev, const infinicore::DataType &dt) const;

    mutable std::optional<infinicore::Tensor> w1_stacked_;
    mutable std::optional<infinicore::Tensor> w2_stacked_;
    mutable std::optional<infinicore::Device> stacked_dev_;
    mutable std::optional<infinicore::DataType> stacked_dt_;
};

} // namespace infinilm::models::minicpm5_moe

