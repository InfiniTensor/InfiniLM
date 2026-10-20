#ifndef INFINI_OPS_METAX_BLAS_H_
#define INFINI_OPS_METAX_BLAS_H_

#include <utility>

// clang-format off
#include <mcblas/mcblas.h>
// clang-format on

#include "data_type.h"
#include "native/cuda/blas.h"
#include "native/cuda/metax/blas_utils.h"
#include "native/cuda/metax/runtime_.h"

namespace infini::ops {

template <>
struct Blas<Device::Type::kMetax> : public Runtime<Device::Type::kMetax> {
  using BlasHandle = mcblasHandle_t;

  static constexpr auto BLAS_OP_N = MCBLAS_OP_N;

  static constexpr auto BLAS_OP_T = MCBLAS_OP_T;

  static constexpr auto R_16F = MACA_R_16F;

  static constexpr auto R_16BF = MACA_R_16BF;

  static constexpr auto R_32F = MACA_R_32F;

  static constexpr auto BLAS_COMPUTE_32F = MCBLAS_COMPUTE_32F;

  static constexpr auto BLAS_COMPUTE_32F_FAST_TF32 =
      MCBLAS_COMPUTE_32F_FAST_TF32;

  static constexpr auto BLAS_GEMM_DEFAULT = MCBLAS_GEMM_DEFAULT;

  static constexpr auto BlasCreate = mcblasCreate;

  static constexpr auto BlasSetStream = mcblasSetStream;

  static constexpr auto BlasDestroy = mcblasDestroy;

  static constexpr auto BlasGemmStridedBatchedEx = [](auto&&... args) {
    return mcblasGemmStridedBatchedEx(std::forward<decltype(args)>(args)...);
  };
};

}  // namespace infini::ops

#endif
