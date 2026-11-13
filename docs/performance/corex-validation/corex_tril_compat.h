#ifndef INFINI_OPS_ILUVATAR_TRIL_KERNEL_H_
#define INFINI_OPS_ILUVATAR_TRIL_KERNEL_H_
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/triangular/kernel.h"
namespace infini::ops {
template <>
class Operator<Tril, Device::Type::kIluvatar>
    : public triangular_detail::CudaTriangular<Runtime<Device::Type::kIluvatar>, Tril, false> {
 public:
  using triangular_detail::CudaTriangular<Runtime<Device::Type::kIluvatar>, Tril, false>::CudaTriangular;
};
}
#endif
