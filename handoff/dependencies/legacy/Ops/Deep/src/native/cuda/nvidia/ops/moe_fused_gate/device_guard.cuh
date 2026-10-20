#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
namespace infini::ops {
class MoeDeviceGuard {
 public:
  explicit MoeDeviceGuard(int device) {
    auto s=cudaGetDevice(&previous_); if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));
    if(previous_!=device) { s=cudaSetDevice(device); if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s)); restore_=true; }
  }
  ~MoeDeviceGuard() { if(restore_) { const auto s=cudaSetDevice(previous_); if(s!=cudaSuccess)std::fprintf(stderr,"MoE restore device failed: %s\n",cudaGetErrorString(s)); } }
 private: int previous_=0; bool restore_=false;
};
}
