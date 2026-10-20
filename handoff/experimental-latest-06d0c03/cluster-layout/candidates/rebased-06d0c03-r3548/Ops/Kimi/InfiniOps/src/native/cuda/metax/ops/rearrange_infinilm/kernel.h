#ifndef INFINI_OPS_METAX_REARRANGE_INFINILM_KERNEL_H_
#define INFINI_OPS_METAX_REARRANGE_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/rearrange_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<RearrangeInfinilm, Device::Type::kMetax>
    : public CudaRearrangeInfinilm<Runtime<Device::Type::kMetax>> {
 public:
  using CudaRearrangeInfinilm<
      Runtime<Device::Type::kMetax>>::CudaRearrangeInfinilm;
};

}  // namespace infini::ops

#endif
