#pragma once

// Private CUDA checks shared by both native FlashAttention providers and the
// pinned forward kernels; this header does not depend on either provider.

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

inline void infini_flash_check(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
  }
}

#define INFINI_FLASH_CUDA_CHECK(call) infini_flash_check((call), #call)
#define INFINI_FLASH_KERNEL_CHECK() INFINI_FLASH_CUDA_CHECK(cudaGetLastError())

// Preserve the upstream call sites in the private inference source target.
#define C10_CUDA_CHECK INFINI_FLASH_CUDA_CHECK
#define C10_CUDA_KERNEL_LAUNCH_CHECK INFINI_FLASH_KERNEL_CHECK
