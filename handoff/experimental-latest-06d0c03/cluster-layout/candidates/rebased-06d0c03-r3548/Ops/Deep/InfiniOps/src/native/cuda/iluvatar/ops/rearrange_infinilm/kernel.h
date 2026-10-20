#ifndef INFINI_OPS_ILUVATAR_REARRANGE_INFINILM_KERNEL_H_
#define INFINI_OPS_ILUVATAR_REARRANGE_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/rearrange_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<RearrangeInfinilm, Device::Type::kIluvatar>
    : public CudaRearrangeInfinilm<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaRearrangeInfinilm<
      Runtime<Device::Type::kIluvatar>>::CudaRearrangeInfinilm;
};

}  // namespace infini::ops

#endif
