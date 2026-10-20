#ifndef INFINI_OPS_MOORE_TOPKSOFTMAX_INFINILM_KERNEL_H_
#define INFINI_OPS_MOORE_TOPKSOFTMAX_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/polyfills.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/topksoftmax_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<TopksoftmaxInfinilm, Device::Type::kMoore>
    : public CudaTopksoftmaxInfinilm<Runtime<Device::Type::kMoore>> {
 public:
  using CudaTopksoftmaxInfinilm<
      Runtime<Device::Type::kMoore>>::CudaTopksoftmaxInfinilm;
};

}  // namespace infini::ops

#endif
