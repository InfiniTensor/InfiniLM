#ifndef INFINI_OPS_MARS_GELU_INFINILM_KERNEL_H_
#define INFINI_OPS_MARS_GELU_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/gelu_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<GeluInfinilm, Device::Type::kMars>
    : public CudaGeluInfinilm<Runtime<Device::Type::kMars>> {
 public:
  using CudaGeluInfinilm<Runtime<Device::Type::kMars>>::CudaGeluInfinilm;
};

}  // namespace infini::ops

#endif
