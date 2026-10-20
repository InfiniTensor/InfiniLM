#ifndef INFINI_OPS_METAX_SILU_KERNEL_H_
#define INFINI_OPS_METAX_SILU_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/silu/kernel.h"

namespace infini::ops {

template <>
class Operator<Silu, Device::Type::kMetax>
    : public CudaSilu<Runtime<Device::Type::kMetax>> {
 public:
  using CudaSilu<Runtime<Device::Type::kMetax>>::CudaSilu;
};

}  // namespace infini::ops

#endif
