#ifndef INFINI_OPS_METAX_GEMM_MCBLAS_H_
#define INFINI_OPS_METAX_GEMM_MCBLAS_H_

#include "native/cuda/metax/blas.h"
#include "native/cuda/ops/gemm/blas.h"

namespace infini::ops {

template <>
class Operator<Gemm, Device::Type::kMetax>
    : public BlasGemm<Blas<Device::Type::kMetax>> {
 public:
  using BlasGemm<Blas<Device::Type::kMetax>>::BlasGemm;
};

}  // namespace infini::ops

#endif
