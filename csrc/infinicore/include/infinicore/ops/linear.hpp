#pragma once

#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

Tensor linear(Tensor input, Tensor weight, std::optional<Tensor> bias, float alpha = 1.0f);

void linear_(Tensor out, Tensor input, Tensor weight, std::optional<Tensor> bias, float alpha = 1.0f);

Tensor linear_packed(Tensor input,
                     Tensor packed_weight,
                     std::optional<Tensor> bias,
                     float alpha = 1.0f);

void linear_packed_(Tensor out,
                    Tensor input,
                    Tensor packed_weight,
                    std::optional<Tensor> bias,
                    float alpha = 1.0f);

} // namespace infinicore::op
