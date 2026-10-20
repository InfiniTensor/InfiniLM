#ifndef INFINI_OPS_MOORE_ZEROS_INFINILM_KERNEL_H_
#define INFINI_OPS_MOORE_ZEROS_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/polyfills.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/zeros_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<ZerosInfinilm, Device::Type::kMoore>
    : public CudaZerosInfinilm<Runtime<Device::Type::kMoore>> {
 public:
  using CudaZerosInfinilm<Runtime<Device::Type::kMoore>>::CudaZerosInfinilm;
};

}  // namespace infini::ops

#endif
