#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <utility>

namespace infinicore::op {
INFINICORE_GRAPH_OP_CLASS(AddRMSNorm, Tensor, Tensor, const Tensor &, const Tensor &, const Tensor &, float);

// Fused Add and RMS Normalization
// Returns: (normalized_result, add_result)
// The add_result can be used as residual for subsequent layers
std::pair<Tensor, Tensor> add_rms_norm(const Tensor &a, const Tensor &b, const Tensor &weight, float epsilon = 1e-5f);
void add_rms_norm_(Tensor out, Tensor residual, const Tensor &a, const Tensor &b, const Tensor &weight, float epsilon = 1e-5f);
// Fused Add and RMS Normalization (inplace)
// normalized_result wil be stored in input, add_result will be stored in residual
void add_rms_norm_inplace(Tensor input, Tensor residual, const Tensor &weight, float epsilon = 1e-5f);
} // namespace infinicore::op
