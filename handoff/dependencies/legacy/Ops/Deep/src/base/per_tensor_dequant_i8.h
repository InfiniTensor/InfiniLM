#pragma once
#include <optional>
#include "operator.h"
namespace infini::ops {
class PerTensorDequantI8 : public Operator<PerTensorDequantI8> {
 public:
  static constexpr bool kAutomaticDefault = true;
  PerTensorDequantI8(Tensor x, const Tensor packed, const Tensor scale,
                     const std::optional<Tensor> zero) {}
  virtual void operator()(Tensor x, const Tensor packed, const Tensor scale,
                          const std::optional<Tensor> zero) const = 0;
};
}
