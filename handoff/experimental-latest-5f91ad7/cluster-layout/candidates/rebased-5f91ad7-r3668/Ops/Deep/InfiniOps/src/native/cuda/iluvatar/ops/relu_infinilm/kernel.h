#ifndef INFINI_OPS_ILUVATAR_RELU_INFINILM_KERNEL_H_
#define INFINI_OPS_ILUVATAR_RELU_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/relu_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<ReluInfinilm, Device::Type::kIluvatar>
    : public CudaReluInfinilm<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaReluInfinilm<Runtime<Device::Type::kIluvatar>>::CudaReluInfinilm;
};

}  // namespace infini::ops

#endif
