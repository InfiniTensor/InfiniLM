#ifndef INFINI_OPS_METAX_GELU_INFINILM_KERNEL_H_
#define INFINI_OPS_METAX_GELU_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/gelu_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<GeluInfinilm, Device::Type::kMetax>
    : public CudaGeluInfinilm<Runtime<Device::Type::kMetax>> {
 public:
  using CudaGeluInfinilm<Runtime<Device::Type::kMetax>>::CudaGeluInfinilm;
};

}  // namespace infini::ops

#endif
