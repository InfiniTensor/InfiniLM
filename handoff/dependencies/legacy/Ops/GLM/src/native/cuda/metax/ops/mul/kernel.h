#ifndef INFINI_OPS_METAX_MUL_KERNEL_H_
#define INFINI_OPS_METAX_MUL_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/mul/kernel.h"

namespace infini::ops {

template <>
class Operator<Mul, Device::Type::kMetax>
    : public CudaMul<Runtime<Device::Type::kMetax>> {
 public:
  using CudaMul<Runtime<Device::Type::kMetax>>::CudaMul;
};

}  // namespace infini::ops

#endif
