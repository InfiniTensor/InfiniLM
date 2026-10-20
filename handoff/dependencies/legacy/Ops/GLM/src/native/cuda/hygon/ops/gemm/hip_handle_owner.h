#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <map>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>
namespace infini::ops {
// DTK CUDA compatibility APIs use a null handle for the implicit HIP primary
// context. Do not mix HIP headers into a DTK nvcc translation unit.
class HygonGemmHandle {
 public:
  static std::shared_ptr<HygonGemmHandle> Acquire(int device) {
    auto key=Current();if(key.first!=device)throw std::invalid_argument("GEMM tensor differs from current HIP device");
    static thread_local std::map<std::pair<int,CUcontext>,std::weak_ptr<HygonGemmHandle>> leases;
    for(auto it=leases.begin();it!=leases.end();)if(it->second.expired())it=leases.erase(it);else ++it;
    if(auto it=leases.find(key);it!=leases.end())if(auto live=it->second.lock())return live;
    auto result=std::shared_ptr<HygonGemmHandle>(new HygonGemmHandle(key));leases[key]=result;return result;
  }
  HygonGemmHandle(const HygonGemmHandle&)=delete;
  HygonGemmHandle& operator=(const HygonGemmHandle&)=delete;
  cublasHandle_t get() const {
    if(worker_!=std::this_thread::get_id())throw std::invalid_argument("GEMM handle used by another worker");
    if(Current()!=owner_)throw std::invalid_argument("GEMM handle used on another HIP device/context");
    return handle_;
  }
  ~HygonGemmHandle() noexcept {
    int previous=-1;CUcontext previous_context=nullptr;
    if(cudaGetDevice(&previous)!=cudaSuccess || cuCtxGetCurrent(&previous_context)!=CUDA_SUCCESS)std::terminate();
    if(owner_.second) {if(cuCtxPushCurrent(owner_.second)!=CUDA_SUCCESS)std::terminate();}
    else if(cudaSetDevice(owner_.first)!=cudaSuccess)std::terminate();
    if(cublasDestroy(handle_)!=CUBLAS_STATUS_SUCCESS)std::terminate();
    if(owner_.second) {CUcontext popped=nullptr;if(cuCtxPopCurrent(&popped)!=CUDA_SUCCESS)std::terminate();}
    else {if(cudaSetDevice(previous)!=cudaSuccess)std::terminate();if(previous_context && cuCtxSetCurrent(previous_context)!=CUDA_SUCCESS)std::terminate();}
  }
 private:
  static std::pair<int,CUcontext> Current() {
    int device=-1;CUcontext context=nullptr;
    if(cudaGetDevice(&device)!=cudaSuccess || cuCtxGetCurrent(&context)!=CUDA_SUCCESS)throw std::runtime_error("Cannot query DTK device/context");
    return {device,context};
  }
  explicit HygonGemmHandle(std::pair<int,CUcontext> owner):owner_(owner),worker_(std::this_thread::get_id()) {
    if(cublasCreate(&handle_)!=CUBLAS_STATUS_SUCCESS || !handle_)throw std::runtime_error("Cannot create Hygon BLAS handle");
  }
  std::pair<int,CUcontext> owner_;std::thread::id worker_;cublasHandle_t handle_{nullptr};
};
} // namespace infini::ops
