#ifndef INFINI_OPS_MARS_REARRANGE_INFINILM_KERNEL_H_
#define INFINI_OPS_MARS_REARRANGE_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/rearrange_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<RearrangeInfinilm, Device::Type::kMars>
    : public CudaRearrangeInfinilm<Runtime<Device::Type::kMars>> {
 public:
  using CudaRearrangeInfinilm<
      Runtime<Device::Type::kMars>>::CudaRearrangeInfinilm;
};

}  // namespace infini::ops

#endif
