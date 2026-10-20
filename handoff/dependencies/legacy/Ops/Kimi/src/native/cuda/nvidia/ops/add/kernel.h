#ifndef INFINI_OPS_NVIDIA_ADD_KERNEL_H_
#define INFINI_OPS_NVIDIA_ADD_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/add/kernel.h"

namespace infini::ops {

template <>
class Operator<Add, Device::Type::kNvidia>
    : public CudaAdd<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaAdd<Runtime<Device::Type::kNvidia>>::CudaAdd;
  bool capture_safe() const override {
    return out_type_ == DataType::kFloat16 ||
           out_type_ == DataType::kBFloat16 ||
           out_type_ == DataType::kFloat32 ||
           out_type_ == DataType::kInt32;
  }
};

}  // namespace infini::ops

#endif
