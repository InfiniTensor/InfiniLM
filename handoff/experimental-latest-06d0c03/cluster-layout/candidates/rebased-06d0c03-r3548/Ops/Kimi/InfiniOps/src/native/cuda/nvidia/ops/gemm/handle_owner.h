#ifndef INFINI_OPS_NVIDIA_GEMM_HANDLE_OWNER_H_
#define INFINI_OPS_NVIDIA_GEMM_HANDLE_OWNER_H_

#include <cublas_v2.h>
#include <cuda.h>

#include <cstdio>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace infini::ops {

// Prepared GEMMs (including the existing Call cache) own leases. The registry
// only shares handles within a worker/context; it never keeps a handle alive.
// Keep prepared instances alive until their graphs and in-flight work finish.
// The caller must not reset/destroy the CUDA context while such instances live.
class NvidiaGemmHandle {
 public:
  static std::shared_ptr<NvidiaGemmHandle> Acquire(
      int device, cublasMath_t math_mode = CUBLAS_DEFAULT_MATH) {
    const auto context = CurrentContext();
    CUdevice current_device{};
    const auto status = cuCtxGetDevice(&current_device);
    if (status != CUDA_SUCCESS || current_device != device) {
      throw std::invalid_argument(
          "GEMM tensor device differs from the active CUDA context");
    }
    const auto key = std::make_pair(context, math_mode);
    static thread_local std::map<std::pair<CUcontext, cublasMath_t>,
                                 std::weak_ptr<NvidiaGemmHandle>>
        owners;
    for (auto it = owners.begin(); it != owners.end();) {
      if (it->second.expired())
        it = owners.erase(it);
      else
        ++it;
    }
    if (auto it = owners.find(key); it != owners.end()) {
      if (auto owner = it->second.lock()) return owner;
    }
    auto owner = std::shared_ptr<NvidiaGemmHandle>(
        new NvidiaGemmHandle(context, math_mode));
    owners[key] = owner;
    return owner;
  }

  NvidiaGemmHandle(const NvidiaGemmHandle&) = delete;
  NvidiaGemmHandle& operator=(const NvidiaGemmHandle&) = delete;

  ~NvidiaGemmHandle() noexcept {
    // Destruction may follow a device switch or occur on a cleanup thread.
    // Do not destroy/recreate a handle merely because another device is active.
    const auto push = cuCtxPushCurrent(context_);
    if (push != CUDA_SUCCESS) {
      ReportCleanupError("selecting owning context", static_cast<int>(push));
      return;
    }
    const auto status = cublasDestroy(handle_);
    if (status != CUBLAS_STATUS_SUCCESS) {
      ReportCleanupError("destroying handle", static_cast<int>(status));
    }
    CUcontext popped{};
    const auto pop = cuCtxPopCurrent(&popped);
    if (pop != CUDA_SUCCESS) {
      ReportCleanupError("restoring context", static_cast<int>(pop));
    }
  }

  cublasHandle_t get() const {
    if (worker_ != std::this_thread::get_id()) {
      throw std::invalid_argument(
          "GEMM instance used by a different worker thread");
    }
    if (CurrentContext() != context_) {
      throw std::invalid_argument(
          "GEMM instance used with a different CUDA context");
    }
    return handle_;
  }

 private:
  explicit NvidiaGemmHandle(CUcontext context, cublasMath_t math_mode)
      : context_(context), worker_(std::this_thread::get_id()) {
    const auto status = cublasCreate(&handle_);
    if (status != CUBLAS_STATUS_SUCCESS || handle_ == nullptr) {
      if (handle_ != nullptr) {
        const auto cleanup = cublasDestroy(handle_);
        if (cleanup != CUBLAS_STATUS_SUCCESS) {
          ReportCleanupError("cleaning failed creation",
                             static_cast<int>(cleanup));
        }
      }
      throw std::runtime_error("creating GEMM BLAS handle failed: " +
                               std::to_string(static_cast<int>(status)));
    }
    if (math_mode != CUBLAS_DEFAULT_MATH) {
      const auto math_status = cublasSetMathMode(handle_, math_mode);
      if (math_status != CUBLAS_STATUS_SUCCESS) {
        const auto cleanup = cublasDestroy(handle_);
        handle_ = nullptr;
        if (cleanup != CUBLAS_STATUS_SUCCESS) {
          ReportCleanupError("cleaning failed math mode setup",
                             static_cast<int>(cleanup));
        }
        throw std::runtime_error(
            "setting GEMM BLAS reduction math mode failed: " +
            std::to_string(static_cast<int>(math_status)));
      }
    }
  }

  static CUcontext CurrentContext() {
    CUcontext context{};
    const auto status = cuCtxGetCurrent(&context);
    if (status != CUDA_SUCCESS || context == nullptr) {
      throw std::runtime_error("GEMM requires an active CUDA context: " +
                               std::to_string(static_cast<int>(status)));
    }
    return context;
  }

  static void ReportCleanupError(const char* operation, int status) noexcept {
    std::fprintf(stderr, "InfiniOps GEMM cleanup: %s failed (%d)\n", operation,
                 status);
  }

  CUcontext context_{};
  std::thread::id worker_;
  cublasHandle_t handle_{};
};

}  // namespace infini::ops

#endif
