#ifndef INFINI_OPS_ILUVATAR_GELUTANH_INFINILM_KERNEL_H_
#define INFINI_OPS_ILUVATAR_GELUTANH_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/gelutanh_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<GelutanhInfinilm, Device::Type::kIluvatar>
    : public CudaGelutanhInfinilm<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaGelutanhInfinilm<
      Runtime<Device::Type::kIluvatar>>::CudaGelutanhInfinilm;
};

}  // namespace infini::ops

#endif
