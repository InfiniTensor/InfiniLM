#ifndef INFINI_OPS_MARS_BLAS_H_
#define INFINI_OPS_MARS_BLAS_H_

#include <utility>

// clang-format off
#include <hcblas/hcblas.h>
// clang-format on

#include "data_type.h"
#include "native/cuda/blas.h"
#include "native/cuda/mars/blas_utils.h"
#include "native/cuda/mars/runtime_.h"

namespace infini::ops {

template <>
struct Blas<Device::Type::kMars> : public Runtime<Device::Type::kMars> {
  using BlasHandle = hcblasHandle_t;

  static constexpr auto BLAS_OP_N = HCBLAS_OP_N;

  static constexpr auto BLAS_OP_T = HCBLAS_OP_T;

  static constexpr auto R_16F = HPCC_R_16F;

  static constexpr auto R_16BF = HPCC_R_16BF;

  static constexpr auto R_32F = HPCC_R_32F;

  static constexpr auto BLAS_COMPUTE_32F = HCBLAS_COMPUTE_32F;

  static constexpr auto BLAS_COMPUTE_32F_FAST_TF32 =
      HCBLAS_COMPUTE_32F_FAST_TF32;

  static constexpr auto BLAS_GEMM_DEFAULT = HCBLAS_GEMM_DEFAULT;

  static constexpr auto BlasCreate = hcblasCreate;

  static constexpr auto BlasSetStream = hcblasSetStream;

  static constexpr auto BlasDestroy = hcblasDestroy;

  static constexpr auto BlasGemmStridedBatchedEx = [](auto&&... args) {
    return hcblasGemmStridedBatchedEx(std::forward<decltype(args)>(args)...);
  };
};

}  // namespace infini::ops

#endif
