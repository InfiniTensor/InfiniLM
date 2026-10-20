#ifndef INFINI_OPS_ILUVATAR_BLAS_H_
#define INFINI_OPS_ILUVATAR_BLAS_H_

#include <utility>

// clang-format off
#include "cublas_v2.h"
// clang-format on

#include "data_type.h"
#include "native/cuda/blas.h"
#include "native/cuda/iluvatar/blas_utils.h"
#include "native/cuda/iluvatar/runtime_.h"

namespace infini::ops {

template <>
struct Blas<Device::Type::kIluvatar> : public Runtime<Device::Type::kIluvatar> {
  using BlasHandle = cublasHandle_t;

  static constexpr auto BLAS_OP_N = CUBLAS_OP_N;

  static constexpr auto BLAS_OP_T = CUBLAS_OP_T;

  static constexpr auto R_16F = CUDA_R_16F;

  static constexpr auto R_16BF = CUDA_R_16BF;

  static constexpr auto R_32F = CUDA_R_32F;

  // Iluvatar uses `cudaDataType` for `computeType`, so we use `CUDA_R_32F`
  // instead of `CUBLAS_COMPUTE_32F_FAST_TF32`.
  static constexpr auto BLAS_COMPUTE_32F = CUDA_R_32F;

  static constexpr auto BLAS_COMPUTE_32F_FAST_TF32 = CUDA_R_32F;

  // Iluvatar uses `CUBLAS_GEMM_DEFAULT_TENSOR_OP` instead of
  // `CUBLAS_GEMM_DEFAULT`.
  static constexpr auto BLAS_GEMM_DEFAULT = CUBLAS_GEMM_DEFAULT_TENSOR_OP;

  static constexpr auto BlasCreate = cublasCreate;

  static constexpr auto BlasSetStream = cublasSetStream;

  static constexpr auto BlasDestroy = cublasDestroy;

  static constexpr auto BlasGemmStridedBatchedEx = [](auto&&... args) {
    return cublasGemmStridedBatchedEx(std::forward<decltype(args)>(args)...);
  };
};

}  // namespace infini::ops

#endif
