#ifndef INFINI_OPS_METAX_SIGMOID_INFINILM_KERNEL_H_
#define INFINI_OPS_METAX_SIGMOID_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/sigmoid_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<SigmoidInfinilm, Device::Type::kMetax>
    : public CudaSigmoidInfinilm<Runtime<Device::Type::kMetax>> {
 public:
  using CudaSigmoidInfinilm<Runtime<Device::Type::kMetax>>::CudaSigmoidInfinilm;
};

}  // namespace infini::ops

#endif
