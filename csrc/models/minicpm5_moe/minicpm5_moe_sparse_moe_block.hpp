#pragma once

#include "../../layers/common_modules.hpp"
#include "infinicore/ops/inductor_segment.hpp"

#include <optional>

namespace infinilm::models::minicpm5_moe {

using MiniCPM5MoeMLP = infinilm::layers::MoeMLP;

/**
 * Sparse MoE: Track B uses AOTI (`inductor_moe_` / InductorMoe).
 * CG-1: MoE is an hcGraph **host break** (eager Triton between device segments).
 * Full MoE-in-device capture is Phase 2.
 * CPU router path remains as an unused private helper (fallback disabled this phase).
 */
class MiniCPM5MoeSparseMoeBlock : public infinicore::nn::Module {
public:
    MiniCPM5MoeSparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                              size_t layer_idx,
                              const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

    /// TP=1 stub for piecewise graph / allreduce segment ABI.
    infinicore::Tensor forward_matmul_only(const infinicore::Tensor &hidden_states) const {
        return forward(hidden_states);
    }
    void allreduce_output(infinicore::Tensor &) const {}

    size_t layer_idx() const { return layer_idx_; }

    infinicore::op::inductor_segment_impl::MoeExternalWeightTensors
    moe_external_weights() const;

protected:
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, gate);
    INFINICORE_NN_PARAMETER(e_score_correction_bias);
    INFINICORE_NN_MODULE_VEC(MiniCPM5MoeMLP, experts);
    INFINICORE_NN_MODULE(MiniCPM5MoeMLP, shared_experts);

    size_t layer_idx_{0};

private:
    /// Legacy CPU sparse MoE (not used by ``forward`` while Track B develops).
    infinicore::Tensor forward_cpu_sparse_(const infinicore::Tensor &hidden_states) const;

    mutable std::optional<infinicore::op::inductor_segment_impl::MoeExternalWeightTensors>
        packed_weights_cache_;
};

} // namespace infinilm::models::minicpm5_moe
