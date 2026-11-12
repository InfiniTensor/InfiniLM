#pragma once

#include "infinicore/tensor.hpp"

#include <cstddef>
#include <stdexcept>

namespace infinicore::op {

enum class PiecewiseInductorSegmentId : int {
    PreAttn = 0,
    PostAttn = 1,
    Moe = 2,
};

namespace inductor_segment_impl {

struct PreAttnExternalWeightTensors {
    Tensor ln_weight;
    Tensor qkv_weight;
    Tensor qkv_bias;
};

struct MoeExternalWeightTensors {
    Tensor gate_weight;
    Tensor expert_weight;
};

inline bool has_package(PiecewiseInductorSegmentId, size_t, size_t) {
    return false;
}

inline void run_pre_attn(size_t,
                         size_t,
                         const Tensor &,
                         const Tensor &,
                         const Tensor &,
                         const PreAttnExternalWeightTensors &) {
    throw std::runtime_error("inductor_segment: AOT packages not available on this build");
}

} // namespace inductor_segment_impl

inline void inductor_warmup_pre_attn_bucket(size_t,
                                            size_t,
                                            size_t,
                                            const inductor_segment_impl::PreAttnExternalWeightTensors &) {
    // no-op stub (MetaX AOT trimmed for QY)
}

} // namespace infinicore::op
