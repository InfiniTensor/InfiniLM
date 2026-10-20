#ifndef INFINI_OPS_MOORE_BLAS_UTILS_H_
#define INFINI_OPS_MOORE_BLAS_UTILS_H_

#include <mublas.h>

#include "data_type.h"
#include "native/cuda/blas_utils.h"

namespace infini::ops {

template <>
struct BlasUtils<Device::Type::kMoore> {
  static musaDataType_t GetDataType(DataType dtype) {
    if (dtype == DataType::kFloat16) return MUSA_R_16F;
    if (dtype == DataType::kBFloat16) return MUSA_R_16BF;
    return MUSA_R_32F;
  }

  static mublasComputeType_t GetComputeType(DataType dtype) {
    if (dtype == DataType::kFloat16) return MUBLAS_COMPUTE_16F;
    if (dtype == DataType::kBFloat16) return MUBLAS_COMPUTE_32F;
    return MUBLAS_COMPUTE_32F;
  }
};

}  // namespace infini::ops

#endif
