#ifndef INFINI_OPS_MOORE_SIGMOID_INFINILM_KERNEL_H_
#define INFINI_OPS_MOORE_SIGMOID_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/polyfills.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/sigmoid_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<SigmoidInfinilm, Device::Type::kMoore>
    : public CudaSigmoidInfinilm<Runtime<Device::Type::kMoore>> {
 public:
  using CudaSigmoidInfinilm<Runtime<Device::Type::kMoore>>::CudaSigmoidInfinilm;
};

}  // namespace infini::ops

#endif
