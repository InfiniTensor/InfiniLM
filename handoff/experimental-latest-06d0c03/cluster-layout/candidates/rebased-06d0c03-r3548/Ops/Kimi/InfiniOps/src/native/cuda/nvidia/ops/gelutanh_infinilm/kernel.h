#ifndef INFINI_OPS_NVIDIA_GELUTANH_INFINILM_KERNEL_H_
#define INFINI_OPS_NVIDIA_GELUTANH_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/gelutanh_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<GelutanhInfinilm, Device::Type::kNvidia>
    : public CudaGelutanhInfinilm<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaGelutanhInfinilm<
      Runtime<Device::Type::kNvidia>>::CudaGelutanhInfinilm;
};

}  // namespace infini::ops

#endif
