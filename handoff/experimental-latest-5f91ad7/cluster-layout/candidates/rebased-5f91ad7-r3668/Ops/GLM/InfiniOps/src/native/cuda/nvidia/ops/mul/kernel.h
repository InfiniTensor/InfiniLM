#ifndef INFINI_OPS_NVIDIA_MUL_KERNEL_H_
#define INFINI_OPS_NVIDIA_MUL_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/ops/mul/scalar.h"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/mul/kernel.h"

namespace infini::ops {

template <>
class Operator<Mul, Device::Type::kNvidia>
    : public CudaMul<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaMul<Runtime<Device::Type::kNvidia>>::CudaMul;
  using CudaMul<Runtime<Device::Type::kNvidia>>::operator();
  Operator(const Tensor input, const double alpha, Tensor out)
      : CudaMul<Runtime<Device::Type::kNvidia>>{input, alpha, out},
        scalar_(std::in_place, input, alpha, out) {}
  static bool supports(const Tensor input, const double alpha, Tensor out) {
    return NvidiaScalarMul::supports(input, alpha, out);
  }
  void operator()(const Tensor input, const double alpha,
                  Tensor out) const override {
    if (!scalar_)
      throw std::invalid_argument("Mul prepared for tensor operands");
    scalar_->Run(input, alpha, out,
                 static_cast<cudaStream_t>(stream_ ? stream_ : 0));
  }
  // All dispatched types use kernels and constructor-owned metadata on replay.
  bool capture_safe() const override { return true; }

 private:
  std::optional<NvidiaScalarMul> scalar_;
};

}  // namespace infini::ops

#endif
