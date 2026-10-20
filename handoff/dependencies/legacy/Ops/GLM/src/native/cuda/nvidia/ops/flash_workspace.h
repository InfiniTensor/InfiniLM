#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

namespace infini::ops::native_flash {

constexpr std::size_t kWorkspaceAlignment = 256;

inline std::size_t checked_add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::overflow_error("native FlashAttention workspace size overflow");
  return a + b;
}

inline std::size_t align_workspace(std::size_t bytes) {
  return checked_add(bytes, kWorkspaceAlignment - 1) &
         ~(kWorkspaceAlignment - 1);
}

struct WorkspaceLayout {
  std::size_t size = 0;
  std::size_t append(std::size_t bytes) {
    const auto offset = align_workspace(size);
    size = checked_add(offset, bytes);
    return offset;
  }
  std::size_t required() const { return align_workspace(size); }
};

// A stream-ordered eager lease. Graph callers supply stable external storage.
// This object never owns workspace for a prepared operator's lifetime.
class WorkspaceLease {
 public:
  WorkspaceLease(void* external, std::size_t available, std::size_t required,
                 cudaStream_t stream, int device)
      : data_{external}, stream_{stream} {
    if (external || available) {
      if (!external || available < required)
        throw std::invalid_argument("native FlashAttention workspace is missing or too small");
      if (reinterpret_cast<std::uintptr_t>(external) % kWorkspaceAlignment)
        throw std::invalid_argument("native FlashAttention workspace is not 256-byte aligned");
      return;
    }
    cudaStreamCaptureStatus capture{cudaStreamCaptureStatusNone};
    check(cudaStreamIsCapturing(stream, &capture), "cudaStreamIsCapturing");
    if (capture != cudaStreamCaptureStatusNone)
      throw std::invalid_argument(
          "native FlashAttention graph capture requires stable external workspace");
    const auto status = cudaMallocAsync(&data_, required, stream);
    if (status != cudaSuccess) {
      int supported = 0;
      const auto attr = cudaDeviceGetAttribute(
          &supported, cudaDevAttrMemoryPoolsSupported, device);
      if (attr == cudaSuccess && !supported)
        throw std::runtime_error(
            "native FlashAttention eager workspace requires CUDA async memory pools");
      check(status, "cudaMallocAsync");
    }
    owned_ = true;
  }

  WorkspaceLease(const WorkspaceLease&) = delete;
  WorkspaceLease& operator=(const WorkspaceLease&) = delete;

  ~WorkspaceLease() noexcept {
    if (!owned_) return;
    const auto status = cudaFreeAsync(data_, stream_);
    if (status == cudaSuccess) return;
    // Failure path only: wait for any submitted work before conventional free.
    const auto synchronized = cudaStreamSynchronize(stream_);
    const auto freed = synchronized == cudaSuccess ? cudaFree(data_) : synchronized;
    std::fprintf(stderr,
                 "native FlashAttention async release failed: %s; cleanup: %s\n",
                 cudaGetErrorString(status), cudaGetErrorString(freed));
  }

  void* data() const { return data_; }

  void finish() {
    if (!owned_) return;
    check(cudaFreeAsync(data_, stream_), "cudaFreeAsync");
    owned_ = false;
    data_ = nullptr;
  }

 private:
  static void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
      throw std::runtime_error(std::string("native FlashAttention ") +
                               operation + ": " + cudaGetErrorString(status));
  }
  void* data_ = nullptr;
  cudaStream_t stream_ = nullptr;
  bool owned_ = false;
};

}  // namespace infini::ops::native_flash
