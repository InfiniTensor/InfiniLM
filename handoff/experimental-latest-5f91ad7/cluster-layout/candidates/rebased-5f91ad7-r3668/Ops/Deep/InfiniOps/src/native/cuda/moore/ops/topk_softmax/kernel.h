#ifndef INFINI_OPS_MOORE_TOPK_SOFTMAX_KERNEL_H_
#define INFINI_OPS_MOORE_TOPK_SOFTMAX_KERNEL_H_

#include <utility>

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/polyfills.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/topk_softmax/kernel.h"

namespace infini::ops {

template <>
class Operator<TopkSoftmax, Device::Type::kMoore, 0>
    : public CudaTopkSoftmax<Runtime<Device::Type::kMoore>> {
 public:
  using CudaTopkSoftmax<Runtime<Device::Type::kMoore>>::CudaTopkSoftmax;
};

}  // namespace infini::ops

#endif
