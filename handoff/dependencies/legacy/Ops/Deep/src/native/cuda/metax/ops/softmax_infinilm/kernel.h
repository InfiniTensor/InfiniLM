#ifndef INFINI_OPS_METAX_SOFTMAX_INFINILM_KERNEL_H_
#define INFINI_OPS_METAX_SOFTMAX_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/softmax_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<SoftmaxInfinilm, Device::Type::kMetax>
    : public CudaSoftmaxInfinilm<Runtime<Device::Type::kMetax>> {
 public:
  using CudaSoftmaxInfinilm<Runtime<Device::Type::kMetax>>::CudaSoftmaxInfinilm;
};

}  // namespace infini::ops

#endif
