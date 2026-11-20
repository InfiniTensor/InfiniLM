#pragma once

#include "layers/linear/fused_linear.hpp"
#include "layers/normalization/rmsnorm.hpp"

namespace infinilm::models::deepseek_v4 {

class Compressor : public weights::ModelWeights {
public:
    Compressor(runtime::Runtime &runtime, size_t hidden, size_t head_dim, size_t ratio, double eps);

    // Returns normalized and rotated FP32 rows for packed cache storage.
    runtime::Tensor forward(runtime::Runtime &runtime, const runtime::Tensor &hidden_states,
                            const runtime::Tensor &sources, const runtime::Tensor &state_slots,
                            const runtime::Tensor &state, const runtime::Tensor &positions,
                            const runtime::Tensor &cos_sin,
                            const std::vector<size_t> &query_offsets) const;

    std::vector<std::shared_ptr<weights::ModelWeights>> loading_children() const override;

private:
    size_t head_dim_;
    runtime::Tensor ape_;
    std::shared_ptr<layers::linear::GateUpParallelLinear> projection_;
    std::shared_ptr<layers::RMSNorm> norm_;
};

} // namespace infinilm::models::deepseek_v4
