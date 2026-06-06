#pragma once

#include "infinicore/tensor.hpp"

#include <cstddef>

namespace infinilm::global_state {

enum class PiecewiseCapturePhase {
    None,
    PreAttn,
    EagerAttn,
    PostAttn,
    LmHead,
};

/// Per-layer staging tensors shared between captured pre-attn graphs and eager attn.
struct PiecewiseLayerStaging {
    infinicore::Tensor q_rope;
    infinicore::Tensor k_rope;
    infinicore::Tensor v_rope;
    infinicore::Tensor attn_output;
};

/// Thread-local piecewise prefill state (capture + replay).
struct PiecewisePrefillState {
    PiecewiseCapturePhase phase{PiecewiseCapturePhase::None};
    size_t active_layer{0};
    /// Actual token count for this request (may be < bucket size).
    size_t valid_seq_len{0};
    /// Padded bucket length for the active replay.
    size_t bucket_seq_len{0};
    infinicore::Tensor hidden_states;
    infinicore::Tensor residual;
    std::vector<PiecewiseLayerStaging> layer_staging;
};

} // namespace infinilm::global_state
