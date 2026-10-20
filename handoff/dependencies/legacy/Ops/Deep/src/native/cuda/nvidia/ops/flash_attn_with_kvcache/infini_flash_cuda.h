#pragma once

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
