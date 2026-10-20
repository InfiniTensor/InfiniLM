#ifndef INFINI_OPS_METAX_RELU_KERNEL_H_
#define INFINI_OPS_METAX_RELU_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/relu/kernel.h"

namespace infini::ops {

template <>
class Operator<Relu, Device::Type::kMetax>
    : public CudaRelu<Runtime<Device::Type::kMetax>> {
 public:
  using CudaRelu<Runtime<Device::Type::kMetax>>::CudaRelu;
};

}  // namespace infini::ops

#endif
