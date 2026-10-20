#ifndef INFINI_OPS_METAX_FILL_KERNEL_H_
#define INFINI_OPS_METAX_FILL_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/fill/kernel.h"

namespace infini::ops {

template <>
class Operator<Fill, Device::Type::kMetax>
    : public CudaFill<Runtime<Device::Type::kMetax>> {
 public:
  using CudaFill<Runtime<Device::Type::kMetax>>::CudaFill;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_METAX_FILL_KERNEL_H_
