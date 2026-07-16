#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

Tensor bilinear(Tensor x1, Tensor x2, Tensor weight, std::optional<Tensor> bias);
void bilinear_(Tensor out, Tensor x1, Tensor x2, Tensor weight, std::optional<Tensor> bias);

} // namespace infinicore::op
