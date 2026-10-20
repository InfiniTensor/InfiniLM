#ifndef INFINI_OPS_METAX_TOPKSOFTMAX_INFINILM_KERNEL_H_
#define INFINI_OPS_METAX_TOPKSOFTMAX_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/topksoftmax_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<TopksoftmaxInfinilm, Device::Type::kMetax>
    : public CudaTopksoftmaxInfinilm<Runtime<Device::Type::kMetax>> {
 public:
  using CudaTopksoftmaxInfinilm<
      Runtime<Device::Type::kMetax>>::CudaTopksoftmaxInfinilm;
};

}  // namespace infini::ops

#endif
