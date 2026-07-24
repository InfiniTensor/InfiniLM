#pragma once

#include "common/op.hpp"

namespace infinicore::op {

Tensor cat(std::vector<Tensor> tensors, int dim);
void cat_(Tensor out, std::vector<Tensor> tensors, int dim);
} // namespace infinicore::op
