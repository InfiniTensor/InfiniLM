#ifndef INFINI_OPS_METAX_GELUTANH_INFINILM_KERNEL_H_
#define INFINI_OPS_METAX_GELUTANH_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/gelutanh_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<GelutanhInfinilm, Device::Type::kMetax>
    : public CudaGelutanhInfinilm<Runtime<Device::Type::kMetax>> {
 public:
  using CudaGelutanhInfinilm<
      Runtime<Device::Type::kMetax>>::CudaGelutanhInfinilm;
};

}  // namespace infini::ops

#endif
