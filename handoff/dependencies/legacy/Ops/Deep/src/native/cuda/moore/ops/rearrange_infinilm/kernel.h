#ifndef INFINI_OPS_MOORE_REARRANGE_INFINILM_KERNEL_H_
#define INFINI_OPS_MOORE_REARRANGE_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/polyfills.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/rearrange_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<RearrangeInfinilm, Device::Type::kMoore>
    : public CudaRearrangeInfinilm<Runtime<Device::Type::kMoore>> {
 public:
  using CudaRearrangeInfinilm<
      Runtime<Device::Type::kMoore>>::CudaRearrangeInfinilm;
};

}  // namespace infini::ops

#endif
