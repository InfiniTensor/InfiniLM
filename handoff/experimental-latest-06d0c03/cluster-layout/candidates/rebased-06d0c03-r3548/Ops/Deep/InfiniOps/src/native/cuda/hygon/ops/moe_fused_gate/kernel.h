#pragma once
#include "base/moe_fused_gate.h"
namespace infini::ops {
template <> class Operator<MoeFusedGate, Device::Type::kHygon, 0> : public MoeFusedGate {
 public:
  using MoeFusedGate::MoeFusedGate;
  bool capture_safe() const override { return false; }
  static bool supports(Tensor weights, Tensor indices, const Tensor input, const Tensor bias, size_t groups, size_t selected_groups, size_t shared, float scale, bool apply_scale) {
    const auto e=input.ndim()==2 ? input.size(1) : 0;
    return input.device().type()==Device::Type::kHygon && MoeFusedGate::supports(weights,indices,input,bias,groups,selected_groups,shared,scale,apply_scale) &&
        !(e&(e-1)) && groups<=32 && !(groups&(groups-1)) && e/groups<=32;
  }
  void operator()(Tensor weights, Tensor indices, const Tensor input, const Tensor bias, size_t groups, size_t selected_groups, size_t shared, float scale, bool apply_scale) const override;
};
}
