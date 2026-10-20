#ifndef INFINI_OPS_MARS_TOPKSOFTMAX_INFINILM_KERNEL_H_
#define INFINI_OPS_MARS_TOPKSOFTMAX_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/topksoftmax_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<TopksoftmaxInfinilm, Device::Type::kMars>
    : public CudaTopksoftmaxInfinilm<Runtime<Device::Type::kMars>> {
 public:
  using CudaTopksoftmaxInfinilm<
      Runtime<Device::Type::kMars>>::CudaTopksoftmaxInfinilm;
};

}  // namespace infini::ops

#endif
