#ifndef INFINI_OPS_METAX_SILU_AND_MUL_INFINILM_KERNEL_H_
#define INFINI_OPS_METAX_SILU_AND_MUL_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/silu_and_mul_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<SiluAndMulInfinilm, Device::Type::kMetax>
    : public CudaSiluAndMulInfinilm<Runtime<Device::Type::kMetax>> {
 public:
  using CudaSiluAndMulInfinilm<
      Runtime<Device::Type::kMetax>>::CudaSiluAndMulInfinilm;
};

}  // namespace infini::ops

#endif
