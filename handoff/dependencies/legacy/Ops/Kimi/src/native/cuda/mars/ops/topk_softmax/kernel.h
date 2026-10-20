#ifndef INFINI_OPS_MARS_TOPK_SOFTMAX_KERNEL_H_
#define INFINI_OPS_MARS_TOPK_SOFTMAX_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/topk_softmax/kernel.h"

namespace infini::ops {

template <>
class Operator<TopkSoftmax, Device::Type::kMars, 0>
    : public CudaTopkSoftmax<Runtime<Device::Type::kMars>> {
 public:
  using CudaTopkSoftmax<Runtime<Device::Type::kMars>>::CudaTopkSoftmax;
};

}  // namespace infini::ops

#endif
