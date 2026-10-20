#ifndef INFINI_OPS_ILUVATAR_TOPK_SOFTMAX_KERNEL_H_
#define INFINI_OPS_ILUVATAR_TOPK_SOFTMAX_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/topk_softmax/kernel.h"

namespace infini::ops {

template <>
class Operator<TopkSoftmax, Device::Type::kIluvatar, 0>
    : public CudaTopkSoftmax<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaTopkSoftmax<Runtime<Device::Type::kIluvatar>>::CudaTopkSoftmax;
};

}  // namespace infini::ops

#endif
