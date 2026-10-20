#ifndef INFINI_OPS_NVIDIA_RMS_NORM_KERNEL_H_
#define INFINI_OPS_NVIDIA_RMS_NORM_KERNEL_H_

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/rms_norm/kernel.h"

namespace infini::ops {

template <>
class Operator<RmsNorm, Device::Type::kNvidia>
    : public CudaRmsNorm<Runtime<Device::Type::kNvidia>> {
 public:
  using Base = CudaRmsNorm<Runtime<Device::Type::kNvidia>>;
  using Base::Base;

  // Mixed weight types dispatch to kernels too; replay allocates no metadata.
  bool capture_safe() const override { return true; }

  void operator()(const Tensor input, const Tensor weight, float eps,
                  Tensor out) const override {
    if (out.numel() == 0) return;
    Base::operator()(input, weight, eps, out);
  }
};

}  // namespace infini::ops

#endif
