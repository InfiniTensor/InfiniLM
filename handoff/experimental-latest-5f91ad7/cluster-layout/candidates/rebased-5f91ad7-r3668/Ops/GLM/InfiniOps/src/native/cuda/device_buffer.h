#ifndef INFINI_OPS_CUDA_DEVICE_BUFFER_H_
#define INFINI_OPS_CUDA_DEVICE_BUFFER_H_

#include <cstddef>
#include <exception>
#include <new>
#include <stdexcept>

namespace infini::ops {

// Private ownership for descriptor metadata and scratch allocated at Make.
// Backend keeps this usable by every CUDA-like provider without a CUDA API tie.
template <typename Backend>
class CudaDeviceBuffer {
 public:
  CudaDeviceBuffer() = default;
  CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
  CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;

  std::byte* Allocate(std::size_t bytes) {
    if (!bytes) return nullptr;
    if (data_) throw std::logic_error("device buffer already allocated");
    if (Backend::GetDevice(&owner_device_) != Backend::kSuccess)
      throw std::runtime_error("cannot query allocation owner device");
    void* allocation = nullptr;
    const auto status = Backend::Malloc(&allocation, bytes);
    if (status != Backend::kSuccess || !allocation) {
      if (allocation && Backend::Free(allocation) != Backend::kSuccess)
        std::terminate();
      throw std::bad_alloc();
    }
    data_ = allocation;
    return static_cast<std::byte*>(data_);
  }

  ~CudaDeviceBuffer() {
    if (!data_) return;
    int previous = -1;
    if (Backend::GetDevice(&previous) != Backend::kSuccess) std::terminate();
    const bool switched = previous != owner_device_;
    if (switched && Backend::SetDevice(owner_device_) != Backend::kSuccess)
      std::terminate();
    const auto freed = Backend::Free(data_);
    if (switched && Backend::SetDevice(previous) != Backend::kSuccess)
      std::terminate();
    if (freed != Backend::kSuccess) std::terminate();
  }

 private:
  void* data_{nullptr};
  int owner_device_{-1};
};

}  // namespace infini::ops

#endif
