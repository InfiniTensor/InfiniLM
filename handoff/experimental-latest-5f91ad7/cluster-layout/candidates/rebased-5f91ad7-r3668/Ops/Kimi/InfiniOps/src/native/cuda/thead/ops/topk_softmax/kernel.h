#ifndef INFINI_OPS_THEAD_TOPK_SOFTMAX_KERNEL_H_
#define INFINI_OPS_THEAD_TOPK_SOFTMAX_KERNEL_H_

#include <utility>

#include "native/cuda/ops/topk_softmax/kernel.h"
#include "native/cuda/thead/caster.cuh"
#include "native/cuda/thead/runtime_.h"

namespace infini::ops {

template <>
class Operator<TopkSoftmax, Device::Type::kThead>
    : public CudaTopkSoftmax<Runtime<Device::Type::kThead>> {
 public:
  using CudaTopkSoftmax<Runtime<Device::Type::kThead>>::CudaTopkSoftmax;
};

}  // namespace infini::ops

#endif
