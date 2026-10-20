#ifndef INFINI_OPS_HYGON_TOPK_SOFTMAX_KERNEL_H_
#define INFINI_OPS_HYGON_TOPK_SOFTMAX_KERNEL_H_

#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/topk_softmax/kernel.h"

namespace infini::ops {

template <>
class Operator<TopkSoftmax, Device::Type::kHygon, 0>
    : public CudaTopkSoftmax<Runtime<Device::Type::kHygon>> {
 public:
  using CudaTopkSoftmax<Runtime<Device::Type::kHygon>>::CudaTopkSoftmax;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_HYGON_TOPK_SOFTMAX_KERNEL_H_
