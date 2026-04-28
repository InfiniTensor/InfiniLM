#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(GptqQyblasGemm, Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t);

void scaled_mm_w4a16_gptq_qy_(Tensor out, const Tensor &in, const Tensor &qweight, const Tensor &scales, const Tensor &qzeros, int64_t quant_type, int64_t bit);
} // namespace infinicore::op
