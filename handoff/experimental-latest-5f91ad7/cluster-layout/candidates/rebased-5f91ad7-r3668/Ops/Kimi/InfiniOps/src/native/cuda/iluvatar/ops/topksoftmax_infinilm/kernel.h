#ifndef INFINI_OPS_ILUVATAR_TOPKSOFTMAX_INFINILM_KERNEL_H_
#define INFINI_OPS_ILUVATAR_TOPKSOFTMAX_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/topksoftmax_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<TopksoftmaxInfinilm, Device::Type::kIluvatar>
    : public CudaTopksoftmaxInfinilm<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaTopksoftmaxInfinilm<
      Runtime<Device::Type::kIluvatar>>::CudaTopksoftmaxInfinilm;
};

}  // namespace infini::ops

#endif
