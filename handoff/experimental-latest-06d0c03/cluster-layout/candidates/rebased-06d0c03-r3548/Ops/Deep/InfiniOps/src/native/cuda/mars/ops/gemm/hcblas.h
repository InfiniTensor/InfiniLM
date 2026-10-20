#ifndef INFINI_OPS_MARS_GEMM_HCBLAS_H_
#define INFINI_OPS_MARS_GEMM_HCBLAS_H_

#include "native/cuda/mars/blas.h"
#include "native/cuda/ops/gemm/blas.h"

namespace infini::ops {

template <>
class Operator<Gemm, Device::Type::kMars>
    : public BlasGemm<Blas<Device::Type::kMars>> {
 public:
  using BlasGemm<Blas<Device::Type::kMars>>::BlasGemm;
};

}  // namespace infini::ops

#endif
