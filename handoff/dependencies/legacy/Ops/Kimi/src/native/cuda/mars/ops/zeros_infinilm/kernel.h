#ifndef INFINI_OPS_MARS_ZEROS_INFINILM_KERNEL_H_
#define INFINI_OPS_MARS_ZEROS_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/zeros_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<ZerosInfinilm, Device::Type::kMars>
    : public CudaZerosInfinilm<Runtime<Device::Type::kMars>> {
 public:
  using CudaZerosInfinilm<Runtime<Device::Type::kMars>>::CudaZerosInfinilm;
};

}  // namespace infini::ops

#endif
