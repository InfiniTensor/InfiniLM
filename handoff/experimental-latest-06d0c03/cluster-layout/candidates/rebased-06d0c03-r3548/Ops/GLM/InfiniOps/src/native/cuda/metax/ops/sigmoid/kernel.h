#ifndef INFINI_OPS_METAX_SIGMOID_KERNEL_H_
#define INFINI_OPS_METAX_SIGMOID_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/sigmoid/kernel.h"

namespace infini::ops {

template <>
class Operator<Sigmoid, Device::Type::kMetax>
    : public CudaSigmoid<Runtime<Device::Type::kMetax>> {
 public:
  using CudaSigmoid<Runtime<Device::Type::kMetax>>::CudaSigmoid;
};

}  // namespace infini::ops

#endif
