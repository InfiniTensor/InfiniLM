#ifndef INFINI_OPS_METAX_TOPK_SOFTMAX_KERNEL_H_
#define INFINI_OPS_METAX_TOPK_SOFTMAX_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/topk_softmax/kernel.h"

namespace infini::ops {

template <>
class Operator<TopkSoftmax, Device::Type::kMetax, 0>
    : public CudaTopkSoftmax<Runtime<Device::Type::kMetax>> {
 public:
  using CudaTopkSoftmax<Runtime<Device::Type::kMetax>>::CudaTopkSoftmax;
};

}  // namespace infini::ops

#endif
