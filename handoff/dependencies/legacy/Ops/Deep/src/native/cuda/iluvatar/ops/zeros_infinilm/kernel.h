#ifndef INFINI_OPS_ILUVATAR_ZEROS_INFINILM_KERNEL_H_
#define INFINI_OPS_ILUVATAR_ZEROS_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/zeros_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<ZerosInfinilm, Device::Type::kIluvatar>
    : public CudaZerosInfinilm<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaZerosInfinilm<Runtime<Device::Type::kIluvatar>>::CudaZerosInfinilm;
};

}  // namespace infini::ops

#endif
