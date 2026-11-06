#ifndef INFINI_OPS_ILUVATAR_TRIU_KERNEL_H_
#define INFINI_OPS_ILUVATAR_TRIU_KERNEL_H_
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/triangular/kernel.h"
namespace infini::ops {
template <>
class Operator<Triu, Device::Type::kIluvatar>
    : public triangular_detail::CudaTriangular<Runtime<Device::Type::kIluvatar>, Triu, true> {
 public:
  using triangular_detail::CudaTriangular<Runtime<Device::Type::kIluvatar>, Triu, true>::CudaTriangular;
};
}
#endif
