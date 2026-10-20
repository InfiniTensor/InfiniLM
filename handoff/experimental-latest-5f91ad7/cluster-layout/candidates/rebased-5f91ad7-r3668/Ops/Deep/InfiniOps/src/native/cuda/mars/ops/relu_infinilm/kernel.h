#ifndef INFINI_OPS_MARS_RELU_INFINILM_KERNEL_H_
#define INFINI_OPS_MARS_RELU_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/relu_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<ReluInfinilm, Device::Type::kMars>
    : public CudaReluInfinilm<Runtime<Device::Type::kMars>> {
 public:
  using CudaReluInfinilm<Runtime<Device::Type::kMars>>::CudaReluInfinilm;
};

}  // namespace infini::ops

#endif
