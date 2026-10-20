#ifndef INFINI_OPS_NVIDIA_MUL_KERNEL_H_
#define INFINI_OPS_NVIDIA_MUL_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/mul/kernel.h"

namespace infini::ops {

template <>
class Operator<Mul, Device::Type::kNvidia>
    : public CudaMul<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaMul<Runtime<Device::Type::kNvidia>>::CudaMul;
  bool capture_safe() const override {
    return out_type_ == DataType::kFloat16 ||
           out_type_ == DataType::kBFloat16 ||
           out_type_ == DataType::kFloat32;
  }
};

}  // namespace infini::ops

#endif
