#ifndef INFINI_OPS_THEAD_BLAS_H_
#define INFINI_OPS_THEAD_BLAS_H_

#include <utility>

// clang-format off
#include <cublas_v2.h>
// clang-format on

#include "data_type.h"
#include "native/cuda/blas.h"
#include "native/cuda/thead/blas_utils.h"
#include "native/cuda/thead/runtime_.h"

namespace infini::ops {

template <>
struct Blas<Device::Type::kThead> : public Runtime<Device::Type::kThead> {
  using BlasHandle = cublasHandle_t;

  static constexpr auto BLAS_OP_N = CUBLAS_OP_N;
  static constexpr auto BLAS_OP_T = CUBLAS_OP_T;
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
