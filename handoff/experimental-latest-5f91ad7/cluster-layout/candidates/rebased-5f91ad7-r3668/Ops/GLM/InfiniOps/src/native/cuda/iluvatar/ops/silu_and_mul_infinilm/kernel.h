#ifndef INFINI_OPS_ILUVATAR_SILU_AND_MUL_INFINILM_KERNEL_H_
#define INFINI_OPS_ILUVATAR_SILU_AND_MUL_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/silu_and_mul_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<SiluAndMulInfinilm, Device::Type::kIluvatar>
    : public CudaSiluAndMulInfinilm<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaSiluAndMulInfinilm<
      Runtime<Device::Type::kIluvatar>>::CudaSiluAndMulInfinilm;
};

}  // namespace infini::ops

#endif
