#pragma once
#include <optional>
#include "operator.h"
namespace infini::ops {
class PerTensorQuantI8 : public Operator<PerTensorQuantI8> {
 public:
  static constexpr bool kAutomaticDefault = true;
  PerTensorQuantI8(const Tensor x, Tensor packed, Tensor scale,
                   const std::optional<Tensor> zero, const bool is_static) {}
  virtual void operator()(const Tensor x, Tensor packed, Tensor scale,
                          const std::optional<Tensor> zero,
                          const bool is_static) const = 0;
};
}
