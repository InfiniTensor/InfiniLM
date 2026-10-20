#ifndef INFINI_OPS_MOORE_GEMM_MUBLAS_H_
#define INFINI_OPS_MOORE_GEMM_MUBLAS_H_

#include "native/cuda/moore/blas.h"
#include "native/cuda/ops/gemm/blas.h"

namespace infini::ops {

template <>
class Operator<Gemm, Device::Type::kMoore>
    : public BlasGemm<Blas<Device::Type::kMoore>> {
 public:
  using BlasGemm<Blas<Device::Type::kMoore>>::BlasGemm;

 protected:
  const void* GetAlphaPtr(const float& alpha, DataType dtype) const override {
    if (BlasUtils<Device::Type::kMoore>::GetComputeType(dtype) ==
        MUBLAS_COMPUTE_16F) {
      alpha_fp16_ = Float16::FromFloat(alpha);
      return &alpha_fp16_;
    }

    return &alpha;
  }

  const void* GetBetaPtr(const float& beta, DataType dtype) const override {
    if (BlasUtils<Device::Type::kMoore>::GetComputeType(dtype) ==
        MUBLAS_COMPUTE_16F) {
      beta_fp16_ = Float16::FromFloat(beta);
      return &beta_fp16_;
    }

    return &beta;
  }

 private:
  mutable Float16 alpha_fp16_{};

  mutable Float16 beta_fp16_{};
};

}  // namespace infini::ops

#endif
