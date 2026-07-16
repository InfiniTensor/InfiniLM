#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

Tensor matmul(Tensor a, Tensor b, float alpha = 1.0f);
void matmul_(Tensor c, Tensor a, Tensor b, float alpha = 1.0f);

} // namespace infinicore::op
