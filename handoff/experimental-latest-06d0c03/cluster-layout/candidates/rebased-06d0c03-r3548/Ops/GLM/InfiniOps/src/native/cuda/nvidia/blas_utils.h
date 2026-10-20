#ifndef INFINI_OPS_NVIDIA_BLAS_UTILS_H_
#define INFINI_OPS_NVIDIA_BLAS_UTILS_H_

// clang-format off
#include "cublas_v2.h"
// clang-format on

#include "base/gemm.h"
#include "data_type.h"
#include "native/cuda/blas_utils.h"

namespace infini::ops {

template <>
struct BlasUtils<Device::Type::kNvidia> {
  static auto GetDataType(DataType dtype) {
    if (dtype == DataType::kFloat16) return CUDA_R_16F;
    if (dtype == DataType::kBFloat16) return CUDA_R_16BF;
    return CUDA_R_32F;
  }

  static auto GetComputeType(DataType dtype) {
    if (dtype == DataType::kFloat16 || dtype == DataType::kBFloat16)
      return CUBLAS_COMPUTE_32F;
    return CUBLAS_COMPUTE_32F_FAST_TF32;
  }

  static auto GetComputeType(DataType dtype,
                             GemmConfig::Fp32Precision precision) {
    if (precision == GemmConfig::Fp32Precision::FullFp32)
      return CUBLAS_COMPUTE_32F;
    return GetComputeType(dtype);
  }
};

}  // namespace infini::ops

#endif
