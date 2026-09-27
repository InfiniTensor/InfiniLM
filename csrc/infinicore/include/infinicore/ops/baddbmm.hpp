#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

Tensor baddbmm(Tensor input, Tensor batch1, Tensor batch2,
               float beta = 1.0f,
               float alpha = 1.0f);
void baddbmm_(Tensor out, Tensor input, Tensor batch1, Tensor batch2,
              float beta = 1.0f,
              float alpha = 1.0f);
} // namespace infinicore::op
