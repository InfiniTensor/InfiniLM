#ifndef INFINI_OPS_NVIDIA_TRIL_KERNEL_H_
#define INFINI_OPS_NVIDIA_TRIL_KERNEL_H_

#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/triangular/kernel.h"

namespace infini::ops {

template <>
class Operator<Tril, Device::Type::kNvidia>
    : public triangular_detail::CudaTriangular<
          Runtime<Device::Type::kNvidia>, Tril, false> {
 public:
  using triangular_detail::CudaTriangular<
      Runtime<Device::Type::kNvidia>, Tril, false>::CudaTriangular;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_TRIL_KERNEL_H_
