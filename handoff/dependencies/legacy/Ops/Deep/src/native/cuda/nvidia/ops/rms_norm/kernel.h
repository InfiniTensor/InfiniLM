#ifndef INFINI_OPS_NVIDIA_RMS_NORM_KERNEL_H_
#define INFINI_OPS_NVIDIA_RMS_NORM_KERNEL_H_

#include <cmath>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/rms_norm/kernel.h"

namespace infini::ops {

template <>
class Operator<RmsNorm, Device::Type::kNvidia>
    : public CudaRmsNorm<Runtime<Device::Type::kNvidia>> {
 public:
  using Base = CudaRmsNorm<Runtime<Device::Type::kNvidia>>;

  Operator(const Tensor input, const Tensor weight, float eps, Tensor out)
      : Base(input, weight, eps, out),
        qualified_(input.numel() != 0 && std::isfinite(eps) && eps >= 0 &&
                   input.dtype() == weight.dtype()) {}

  Operator(const Tensor input, const Tensor weight, Tensor out)
      : Operator(input, weight, 1e-6f, out) {}

  bool capture_safe() const override { return qualified_; }

  void operator()(const Tensor input, const Tensor weight, float eps,
                  Tensor out) const override {
    if (out.numel() == 0) return;
    Base::operator()(input, weight, eps, out);
  }

 private:
  bool qualified_;
};

}  // namespace infini::ops

#endif
