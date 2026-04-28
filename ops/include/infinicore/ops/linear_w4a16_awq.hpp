#pragma once

#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

Tensor linear_w4a16_awq(Tensor input, Tensor weight_packed, Tensor weight_scale, Tensor weight_zeros, std::optional<Tensor> bias);

void linear_w4a16_awq_(Tensor out, Tensor input, Tensor weight_packed, Tensor weight_scale, Tensor weight_zeros, std::optional<Tensor> bias);

} // namespace infinicore::op
