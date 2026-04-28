#pragma once

#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

Tensor linear_w4a16_gptq_qy(Tensor in, Tensor qweight, Tensor qzeros, Tensor scales, int64_t quant_type, int64_t bit);

void linear_w4a16_gptq_qy_(Tensor out, Tensor in, Tensor qweights, Tensor scales, Tensor qzeros, int64_t quant_type, int64_t bit);

} // namespace infinicore::op
