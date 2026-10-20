#ifndef INFINI_OPS_NVIDIA_BLAS_H_
#define INFINI_OPS_NVIDIA_BLAS_H_

#include <utility>

// clang-format off
#include "cublas_v2.h"
// clang-format on

#include "data_type.h"
#include "native/cuda/blas.h"
#include "native/cuda/nvidia/blas_utils.h"
#include "native/cuda/nvidia/runtime_.h"

namespace infini::ops {

template <>
struct Blas<Device::Type::kNvidia> : public Runtime<Device::Type::kNvidia> {
  using BlasHandle = cublasHandle_t;

  static constexpr auto BLAS_OP_N = CUBLAS_OP_N;

  static constexpr auto BLAS_OP_T = CUBLAS_OP_T;

  static constexpr auto R_16F = CUDA_R_16F;

  static constexpr auto R_16BF = CUDA_R_16BF;

  static constexpr auto R_32F = CUDA_R_32F;

  static constexpr auto BLAS_COMPUTE_32F = CUBLAS_COMPUTE_32F;

  static constexpr auto BLAS_COMPUTE_32F_FAST_TF32 =
      CUBLAS_COMPUTE_32F_FAST_TF32;

  static constexpr auto BLAS_GEMM_DEFAULT = CUBLAS_GEMM_DEFAULT;

  static constexpr auto BlasCreate = cublasCreate;

  static constexpr auto BlasSetStream = cublasSetStream;

  static constexpr auto BlasDestroy = cublasDestroy;

  static constexpr auto BlasGemmStridedBatchedEx = [](auto&&... args) {
    return cublasGemmStridedBatchedEx(std::forward<decltype(args)>(args)...);
  };
};

}  // namespace infini::ops

#endif
