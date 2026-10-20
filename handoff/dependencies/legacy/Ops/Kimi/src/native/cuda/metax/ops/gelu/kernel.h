#ifndef INFINI_OPS_METAX_GELU_KERNEL_H_
#define INFINI_OPS_METAX_GELU_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/gelu/kernel.h"

namespace infini::ops {

template <>
class Operator<Gelu, Device::Type::kMetax>
    : public CudaGelu<Runtime<Device::Type::kMetax>> {
 public:
  using CudaGelu<Runtime<Device::Type::kMetax>>::CudaGelu;
};

}  // namespace infini::ops

#endif
