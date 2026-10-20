#ifndef INFINI_OPS_MOORE_RELU_INFINILM_KERNEL_H_
#define INFINI_OPS_MOORE_RELU_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/polyfills.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/relu_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<ReluInfinilm, Device::Type::kMoore>
    : public CudaReluInfinilm<Runtime<Device::Type::kMoore>> {
 public:
  using CudaReluInfinilm<Runtime<Device::Type::kMoore>>::CudaReluInfinilm;
};

}  // namespace infini::ops

#endif
