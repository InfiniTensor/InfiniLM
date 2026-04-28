#pragma once

#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

Tensor linear(Tensor input, Tensor weight, std::optional<Tensor> bias);

void linear_(Tensor out, Tensor input, Tensor weight, std::optional<Tensor> bias);

} // namespace infinicore::op
