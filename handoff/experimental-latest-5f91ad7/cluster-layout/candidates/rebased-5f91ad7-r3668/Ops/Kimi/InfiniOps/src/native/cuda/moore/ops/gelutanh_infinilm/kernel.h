#ifndef INFINI_OPS_MOORE_GELUTANH_INFINILM_KERNEL_H_
#define INFINI_OPS_MOORE_GELUTANH_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/polyfills.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/gelutanh_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<GelutanhInfinilm, Device::Type::kMoore>
    : public CudaGelutanhInfinilm<Runtime<Device::Type::kMoore>> {
 public:
  using CudaGelutanhInfinilm<
      Runtime<Device::Type::kMoore>>::CudaGelutanhInfinilm;
};

}  // namespace infini::ops

#endif
