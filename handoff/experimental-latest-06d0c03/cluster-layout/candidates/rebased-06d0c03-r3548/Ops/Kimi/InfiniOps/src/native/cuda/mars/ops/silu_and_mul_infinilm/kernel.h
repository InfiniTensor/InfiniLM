#ifndef INFINI_OPS_MARS_SILU_AND_MUL_INFINILM_KERNEL_H_
#define INFINI_OPS_MARS_SILU_AND_MUL_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/silu_and_mul_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<SiluAndMulInfinilm, Device::Type::kMars>
    : public CudaSiluAndMulInfinilm<Runtime<Device::Type::kMars>> {
 public:
  using CudaSiluAndMulInfinilm<
      Runtime<Device::Type::kMars>>::CudaSiluAndMulInfinilm;
};

}  // namespace infini::ops

#endif
