#ifndef INFINI_OPS_MOORE_GELU_INFINILM_KERNEL_H_
#define INFINI_OPS_MOORE_GELU_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/polyfills.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/gelu_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<GeluInfinilm, Device::Type::kMoore>
    : public CudaGeluInfinilm<Runtime<Device::Type::kMoore>> {
 public:
  using CudaGeluInfinilm<Runtime<Device::Type::kMoore>>::CudaGeluInfinilm;
};

}  // namespace infini::ops

#endif
