#ifndef INFINI_CCL_DEVICES_CPU_RUNTIME_H_
#define INFINI_CCL_DEVICES_CPU_RUNTIME_H_

#include <cstdlib>
#include <cstring>

#include "return_status_impl.h"
#include "runtime.h"

namespace infini::ccl {

template <>
struct Runtime<Device::Type::kCpu> : RuntimeBase<Runtime<Device::Type::kCpu>> {
  using Stream = void *;

  static constexpr Device::Type kDeviceType = Device::Type::kCpu;

  static constexpr auto Check = [](auto &&...) {
    return ReturnStatus::kSuccess;
  };

  static constexpr auto Malloc = [](void **ptr, std::size_t size) {
    *ptr = std::malloc(size);
    return *ptr ? ReturnStatus::kSuccess : ReturnStatus::kSystemError;
  };

  static constexpr auto Free = [](void *ptr) {
    std::free(ptr);
    return ReturnStatus::kSuccess;
  };

  static constexpr auto Memcpy = [](void *dst, const void *src,
                                    std::size_t size, int /*kind*/) {
    if (size > 0 && (dst == nullptr || src == nullptr)) {
      return ReturnStatus::kSystemError;
    }
    std::memcpy(dst, src, size);
    return ReturnStatus::kSuccess;
  };

  static constexpr auto Memset = [](void *ptr, int value, std::size_t size) {
    std::memset(ptr, value, size);
    return ReturnStatus::kSuccess;
  };

  static constexpr int MemcpyHostToDevice = 0;

  static constexpr int MemcpyDeviceToHost = 1;

  static constexpr auto GetDevice = [](int *dev) {
    *dev = 0;
    return ReturnStatus::kSuccess;
  };

  static constexpr auto SetDevice = [](int) { return ReturnStatus::kSuccess; };

  static constexpr auto DeviceSynchronize = []() {
    return ReturnStatus::kSuccess;
  };

  static constexpr auto StreamSynchronize = [](Stream) {
    return ReturnStatus::kSuccess;
  };
};

static_assert(Runtime<Device::Type::kCpu>::Validate());

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_CPU_RUNTIME_H_
