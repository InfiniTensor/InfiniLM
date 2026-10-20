#ifndef INFINI_OPS_METAX_BLAS_UTILS_H_
#define INFINI_OPS_METAX_BLAS_UTILS_H_

// clang-format off
#include <mcblas/mcblas.h>
// clang-format on

#include "data_type.h"
#include "native/cuda/blas_utils.h"

namespace infini::ops {

template <>
struct BlasUtils<Device::Type::kMetax> {
  static auto GetDataType(DataType dtype) {
    if (dtype == DataType::kFloat16) return MACA_R_16F;
    if (dtype == DataType::kBFloat16) return MACA_R_16BF;
    return MACA_R_32F;
  }

  static auto GetComputeType(DataType dtype) {
    if (dtype == DataType::kFloat16 || dtype == DataType::kBFloat16)
      return MCBLAS_COMPUTE_32F;
    return MCBLAS_COMPUTE_32F_FAST_TF32;
  }
};

}  // namespace infini::ops

#endif
