#pragma once
#include "base/moe_compute_common.h"
namespace infini::ops {
class MoeFusedGate : public Operator<MoeFusedGate> {
 public:
  MoeFusedGate(Tensor weights, Tensor indices, const Tensor input,
               const Tensor bias, size_t groups, size_t selected_groups,
               size_t shared, float scale, bool apply_scale)
      : metadata_{weights, indices, input, bias},
        groups_(groups),
        selected_groups_(selected_groups),
        shared_(shared),
        scale_(scale),
        apply_scale_(apply_scale) {
    Validate(weights, indices, input, bias, groups, selected_groups, shared,
             scale, apply_scale);
  }
  static bool supports(Tensor weights, Tensor indices, const Tensor input,
                       const Tensor bias, size_t groups, size_t selected_groups,
                       size_t shared, float scale, bool apply_scale) {
    try {
      Validate(weights, indices, input, bias, groups, selected_groups, shared,
               scale, apply_scale);
      return true;
    } catch (const std::invalid_argument&) {
      return false;
    }
  }
  virtual void operator()(Tensor weights, Tensor indices, const Tensor input,
                          const Tensor bias, size_t groups,
                          size_t selected_groups, size_t shared, float scale,
                          bool apply_scale) const = 0;

 protected:
  void Invocation(Tensor weights, Tensor indices, const Tensor input,
                  const Tensor bias, size_t groups, size_t selected_groups,
                  size_t shared, float scale, bool apply_scale) const {
    moe_detail::Require(
        groups == groups_ && selected_groups == selected_groups_ &&
        shared == shared_ && scale == scale_ && apply_scale == apply_scale_);
    moe_detail::Same(metadata_, {weights, indices, input, bias});
    Validate(weights, indices, input, bias, groups, selected_groups, shared,
             scale, apply_scale);
  }
  static void Validate(Tensor weights, Tensor indices, const Tensor input,
                       const Tensor bias, size_t groups, size_t selected_groups,
                       size_t shared, float scale, bool apply_scale) {
    using namespace moe_detail;
    Require(input.ndim() == 2 && weights.ndim() == 2 && indices.ndim() == 2 &&
            bias.ndim() == 1);
    Require(input.dtype() == DataType::kFloat32 || Reduced(input.dtype()));
    Require(weights.dtype() == DataType::kFloat32 &&
            bias.dtype() == DataType::kFloat32 &&
            indices.dtype() == DataType::kInt32);
    const auto n = input.size(0), e = input.size(1), k = weights.size(1);
    Bound(n);
    Bound(e);
    Bound(k);
    Bound(Product(n, k));
    Require(e && groups && e % groups == 0 && e / groups >= 2);
    Require(selected_groups && selected_groups <= groups && k > shared &&
            k - shared <= selected_groups * (e / groups));
    Require(std::isfinite(scale) && scale > 0 && weights.size(0) == n &&
            indices.shape() == weights.shape() && bias.size(0) == e);
    Views({weights, indices, input, bias}, 2);
  }
  std::vector<Tensor> metadata_;
  size_t groups_, selected_groups_, shared_;
  float scale_;
  bool apply_scale_;
};
}  // namespace infini::ops
