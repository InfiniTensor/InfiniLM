#ifndef INFINI_OPS_MOORE_SILU_AND_MUL_INFINILM_KERNEL_H_
#define INFINI_OPS_MOORE_SILU_AND_MUL_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/polyfills.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/silu_and_mul_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<SiluAndMulInfinilm, Device::Type::kMoore>
    : public CudaSiluAndMulInfinilm<Runtime<Device::Type::kMoore>> {
 public:
  using CudaSiluAndMulInfinilm<
      Runtime<Device::Type::kMoore>>::CudaSiluAndMulInfinilm;
};

}  // namespace infini::ops

#endif
