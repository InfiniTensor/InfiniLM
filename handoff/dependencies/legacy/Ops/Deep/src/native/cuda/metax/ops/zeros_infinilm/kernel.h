#ifndef INFINI_OPS_METAX_ZEROS_INFINILM_KERNEL_H_
#define INFINI_OPS_METAX_ZEROS_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/zeros_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<ZerosInfinilm, Device::Type::kMetax>
    : public CudaZerosInfinilm<Runtime<Device::Type::kMetax>> {
 public:
  using CudaZerosInfinilm<Runtime<Device::Type::kMetax>>::CudaZerosInfinilm;
};

}  // namespace infini::ops

#endif
