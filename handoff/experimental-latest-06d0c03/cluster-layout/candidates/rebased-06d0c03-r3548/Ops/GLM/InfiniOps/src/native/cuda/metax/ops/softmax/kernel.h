#ifndef INFINI_OPS_METAX_SOFTMAX_KERNEL_H_
#define INFINI_OPS_METAX_SOFTMAX_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/softmax/kernel.h"

namespace infini::ops {

template <>
class Operator<Softmax, Device::Type::kMetax>
    : public CudaSoftmax<Runtime<Device::Type::kMetax>> {
 public:
  using CudaSoftmax<Runtime<Device::Type::kMetax>>::CudaSoftmax;
};

}  // namespace infini::ops

#endif
