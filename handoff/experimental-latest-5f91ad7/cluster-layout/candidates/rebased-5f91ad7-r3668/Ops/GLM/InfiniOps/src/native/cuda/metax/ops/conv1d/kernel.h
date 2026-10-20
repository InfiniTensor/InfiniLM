#ifndef INFINI_OPS_METAX_CONV1D_KERNEL_H_
#define INFINI_OPS_METAX_CONV1D_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/convolution/kernel.h"

namespace infini::ops {

template <>
class Operator<Conv1d, Device::Type::kMetax>
    : public CudaConv<Runtime<Device::Type::kMetax>, Conv1d> {
 public:
  using CudaConv<Runtime<Device::Type::kMetax>, Conv1d>::CudaConv;
};

}  // namespace infini::ops

#endif
