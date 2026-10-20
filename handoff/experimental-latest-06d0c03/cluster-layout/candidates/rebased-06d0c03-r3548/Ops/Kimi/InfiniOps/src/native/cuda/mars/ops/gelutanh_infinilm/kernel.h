#ifndef INFINI_OPS_MARS_GELUTANH_INFINILM_KERNEL_H_
#define INFINI_OPS_MARS_GELUTANH_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/gelutanh_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<GelutanhInfinilm, Device::Type::kMars>
    : public CudaGelutanhInfinilm<Runtime<Device::Type::kMars>> {
 public:
  using CudaGelutanhInfinilm<
      Runtime<Device::Type::kMars>>::CudaGelutanhInfinilm;
};

}  // namespace infini::ops

#endif
