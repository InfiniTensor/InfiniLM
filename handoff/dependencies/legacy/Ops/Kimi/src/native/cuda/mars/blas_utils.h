#ifndef INFINI_OPS_MARS_BLAS_UTILS_H_
#define INFINI_OPS_MARS_BLAS_UTILS_H_

// clang-format off
#include <hcblas/hcblas.h>
// clang-format on

#include "data_type.h"
#include "native/cuda/blas_utils.h"

namespace infini::ops {

template <>
struct BlasUtils<Device::Type::kMars> {
  static auto GetDataType(DataType dtype) {
    if (dtype == DataType::kFloat16) return HPCC_R_16F;
    if (dtype == DataType::kBFloat16) return HPCC_R_16BF;
    return HPCC_R_32F;
  }

  static auto GetComputeType(DataType dtype) {
    if (dtype == DataType::kFloat16 || dtype == DataType::kBFloat16)
      return HCBLAS_COMPUTE_32F;
    return HCBLAS_COMPUTE_32F_FAST_TF32;
  }
};

}  // namespace infini::ops

#endif
