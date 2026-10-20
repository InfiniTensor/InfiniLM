#ifndef INFINI_OPS_MOORE_BLAS_H_
#define INFINI_OPS_MOORE_BLAS_H_

#include <mublas.h>

#include <utility>

#include "data_type.h"
#include "native/cuda/blas.h"
#include "native/cuda/moore/blas_utils.h"
#include "native/cuda/moore/runtime_.h"

namespace infini::ops {

template <>
struct Blas<Device::Type::kMoore> : public Runtime<Device::Type::kMoore> {
  using BlasHandle = mublasHandle_t;

  static constexpr auto BLAS_OP_N = MUBLAS_OP_N;

  static constexpr auto BLAS_OP_T = MUBLAS_OP_T;

  static constexpr auto R_16F = MUSA_R_16F;

  static constexpr auto R_16BF = MUSA_R_16BF;

  static constexpr auto R_32F = MUSA_R_32F;

  static constexpr auto BLAS_GEMM_DEFAULT = MUBLAS_GEMM_DEFAULT;

  static constexpr auto BlasCreate = mublasCreate;

  static constexpr auto BlasSetStream = mublasSetStream;

  static constexpr auto BlasDestroy = mublasDestroy;

  static constexpr auto BlasGemmStridedBatchedEx = [](auto&&... args) {
    return mublasGemmStridedBatchedEx(std::forward<decltype(args)>(args)...);
  };
};

}  // namespace infini::ops

#endif
