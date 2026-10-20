#ifndef INFINI_OPS_HYGON_BLAS_UTILS_H_
#define INFINI_OPS_HYGON_BLAS_UTILS_H_

// clang-format off
#include "cublas_v2.h"
// clang-format on

#include "data_type.h"
#include "native/cuda/blas_utils.h"

namespace infini::ops {

template <>
struct BlasUtils<Device::Type::kHygon> {
  static auto GetDataType(DataType dtype) {
    if (dtype == DataType::kFloat16) return CUDA_R_16F;
    if (dtype == DataType::kBFloat16) return CUDA_R_16BF;
    return CUDA_R_32F;
  }

  static auto GetComputeType(DataType dtype) {
    (void)dtype;
    return CUBLAS_COMPUTE_32F;
  }
};

}  // namespace infini::ops

#endif
