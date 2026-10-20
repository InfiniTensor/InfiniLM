#ifndef INFINI_OPS_MOORE_SOFTMAX_INFINILM_KERNEL_H_
#define INFINI_OPS_MOORE_SOFTMAX_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/polyfills.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/softmax_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<SoftmaxInfinilm, Device::Type::kMoore>
    : public CudaSoftmaxInfinilm<Runtime<Device::Type::kMoore>> {
 public:
  using CudaSoftmaxInfinilm<Runtime<Device::Type::kMoore>>::CudaSoftmaxInfinilm;
};

}  // namespace infini::ops

#endif
