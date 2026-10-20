#ifndef INFINI_OPS_ILUVATAR_GEMM_CUBLAS_H_
#define INFINI_OPS_ILUVATAR_GEMM_CUBLAS_H_

#include "native/cuda/iluvatar/blas.h"
#include "native/cuda/ops/gemm/blas.h"

namespace infini::ops {

template <>
class Operator<Gemm, Device::Type::kIluvatar>
    : public BlasGemm<Blas<Device::Type::kIluvatar>> {
 public:
  using BlasGemm<Blas<Device::Type::kIluvatar>>::BlasGemm;
};

}  // namespace infini::ops

#endif
