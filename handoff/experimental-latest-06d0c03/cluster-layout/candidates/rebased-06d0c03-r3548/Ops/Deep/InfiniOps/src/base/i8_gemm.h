#pragma once
#include <optional>

#include "operator.h"
namespace infini::ops {
class I8Gemm : public Operator<I8Gemm> {
 public:
  static constexpr bool kAutomaticDefault = true;
  I8Gemm(Tensor c, const Tensor a, const Tensor as, const Tensor b,
         const Tensor bs, const std::optional<Tensor> bias) {}
  virtual void operator()(Tensor c, const Tensor a, const Tensor as,
                          const Tensor b, const Tensor bs,
                          const std::optional<Tensor> bias) const = 0;
};
}  // namespace infini::ops
