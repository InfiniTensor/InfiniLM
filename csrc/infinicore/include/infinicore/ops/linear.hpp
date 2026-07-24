#pragma once

#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

Tensor linear(Tensor input, Tensor weight, std::optional<Tensor> bias, float alpha = 1.0f);

void linear_(Tensor out, Tensor input, Tensor weight, std::optional<Tensor> bias, float alpha = 1.0f);

} // namespace infinicore::op
