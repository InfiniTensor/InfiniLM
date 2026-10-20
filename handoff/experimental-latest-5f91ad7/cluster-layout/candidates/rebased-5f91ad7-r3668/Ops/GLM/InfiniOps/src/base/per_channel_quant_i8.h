#pragma once
#include "operator.h"
namespace infini::ops {
class PerChannelQuantI8 : public Operator<PerChannelQuantI8> {
 public:
  static constexpr bool kAutomaticDefault = true;
  PerChannelQuantI8(const Tensor x, Tensor packed, Tensor scale) {}
  virtual void operator()(const Tensor x, Tensor packed,
                          Tensor scale) const = 0;
};
}  // namespace infini::ops
