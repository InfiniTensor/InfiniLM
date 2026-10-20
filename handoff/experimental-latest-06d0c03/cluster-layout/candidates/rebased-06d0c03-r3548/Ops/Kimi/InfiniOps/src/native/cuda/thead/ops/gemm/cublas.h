#ifndef INFINI_OPS_THEAD_GEMM_CUBLAS_H_
#define INFINI_OPS_THEAD_GEMM_CUBLAS_H_

#include "native/cuda/ops/gemm/blas.h"
#include "native/cuda/thead/blas.h"

namespace infini::ops {

template <>
class Operator<Gemm, Device::Type::kThead, 0>
    : public BlasGemm<Blas<Device::Type::kThead>> {
 public:
  using BlasGemm<Blas<Device::Type::kThead>>::BlasGemm;
};

}  // namespace infini::ops

#endif
