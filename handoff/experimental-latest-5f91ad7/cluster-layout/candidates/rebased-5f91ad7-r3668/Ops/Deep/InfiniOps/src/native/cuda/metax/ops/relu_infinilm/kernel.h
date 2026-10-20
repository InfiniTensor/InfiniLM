#ifndef INFINI_OPS_METAX_RELU_INFINILM_KERNEL_H_
#define INFINI_OPS_METAX_RELU_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/relu_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<ReluInfinilm, Device::Type::kMetax>
    : public CudaReluInfinilm<Runtime<Device::Type::kMetax>> {
 public:
  using CudaReluInfinilm<Runtime<Device::Type::kMetax>>::CudaReluInfinilm;
};

}  // namespace infini::ops

#endif
