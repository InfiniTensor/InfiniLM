#ifndef INFINI_CCL_DEVICES_CUDA_CHECKS_H_
#define INFINI_CCL_DEVICES_CUDA_CHECKS_H_

#include <cuda_runtime.h>

#include <iostream>

#include "return_status_impl.h"

#define INFINI_CHECK_CUDA(result) \
  ::infini::ccl::detail::CheckCudaImpl((result), __FILE__, __LINE__)

namespace infini::ccl {

namespace detail {

inline ReturnStatus CheckCudaImpl(cudaError_t cuda_result, const char *file,
                                  int line) {
  if (cuda_result != cudaSuccess) {
    cudaGetLastError();
    std::cerr << "CUDA error code: " << cuda_result << " at line " << line
              << " in " << file << std::endl;
    std::abort();
  }
  return ReturnStatus::kSuccess;
}

}  // namespace detail

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_CUDA_CHECKS_H_
