#ifndef INFINI_OPS_ILUVATAR_GELU_INFINILM_KERNEL_H_
#define INFINI_OPS_ILUVATAR_GELU_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/gelu_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<GeluInfinilm, Device::Type::kIluvatar>
    : public CudaGeluInfinilm<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaGeluInfinilm<Runtime<Device::Type::kIluvatar>>::CudaGeluInfinilm;
};

}  // namespace infini::ops

#endif
