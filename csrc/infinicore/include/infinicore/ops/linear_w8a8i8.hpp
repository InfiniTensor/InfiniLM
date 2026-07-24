#pragma once

#include "../graph/graph.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

Tensor linear_w8a8i8(Tensor input, Tensor weight_packed, Tensor weight_scale, std::optional<Tensor> bias);

void linear_w8a8i8_(Tensor out, Tensor input, Tensor weight_packed, Tensor weight_scale, std::optional<Tensor> bias);

} // namespace infinicore::op
