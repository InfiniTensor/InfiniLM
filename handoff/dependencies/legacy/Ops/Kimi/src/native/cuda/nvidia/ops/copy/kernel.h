#ifndef INFINI_OPS_NVIDIA_COPY_KERNEL_H_
#define INFINI_OPS_NVIDIA_COPY_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/copy/kernel.h"

namespace infini::ops {

template <>
class Operator<Copy, Device::Type::kNvidia>
    : public CudaCopy<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaCopy<Runtime<Device::Type::kNvidia>>::CudaCopy;
  bool capture_safe() const override {
    return output_size_ != 0 && ndim_ != 0 &&
           (out_type_ == DataType::kFloat32 || out_type_ == DataType::kFloat16 ||
            out_type_ == DataType::kBFloat16 || out_type_ == DataType::kInt32);
  }
};

}  // namespace infini::ops

#endif
