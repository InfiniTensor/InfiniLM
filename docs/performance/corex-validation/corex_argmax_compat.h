#ifndef INFINI_OPS_ILUVATAR_ARGMAX_KERNEL_H_
#define INFINI_OPS_ILUVATAR_ARGMAX_KERNEL_H_
#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/argmax/kernel.h"
namespace infini::ops {
template <>
class Operator<Argmax, Device::Type::kIluvatar>
    : public CudaArgmax<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaArgmax<Runtime<Device::Type::kIluvatar>>::CudaArgmax;
};
}
#endif
