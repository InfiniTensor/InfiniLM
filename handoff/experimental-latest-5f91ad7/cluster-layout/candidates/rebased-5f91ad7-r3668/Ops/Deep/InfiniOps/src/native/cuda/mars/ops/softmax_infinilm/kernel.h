#ifndef INFINI_OPS_MARS_SOFTMAX_INFINILM_KERNEL_H_
#define INFINI_OPS_MARS_SOFTMAX_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/softmax_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<SoftmaxInfinilm, Device::Type::kMars>
    : public CudaSoftmaxInfinilm<Runtime<Device::Type::kMars>> {
 public:
  using CudaSoftmaxInfinilm<Runtime<Device::Type::kMars>>::CudaSoftmaxInfinilm;
};

}  // namespace infini::ops

#endif
