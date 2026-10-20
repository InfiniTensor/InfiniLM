#ifndef INFINI_CCL_DEVICES_MOORE_RUNTIME_H_
#define INFINI_CCL_DEVICES_MOORE_RUNTIME_H_

#include <utility>

// clang-format off
#include <musa_runtime.h>
// clang-format on

#include "devices/cuda/runtime_.h"
#include "devices/moore/device_.h"
#include "logging.h"
#include "return_status_impl.h"

namespace infini::ccl {

template <>
struct Runtime<Device::Type::kMoore>
    : CudaRuntime<Runtime<Device::Type::kMoore>> {
  using Stream = musaStream_t;

  static constexpr Device::Type kDeviceType = Device::Type::kMoore;

  static constexpr auto Check =
      [](auto status, ReturnStatus err_code = ReturnStatus::kSystemError) {
        if (status != musaSuccess) {
          LOG(musaGetErrorString(static_cast<musaError_t>(status)));
          return err_code;
        }
        return ReturnStatus::kSuccess;
      };

  static constexpr auto Malloc = [](auto &&...args) {
    return musaMalloc(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Memcpy = musaMemcpy;

  static constexpr auto Free = [](auto &&...args) {
    return musaFree(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MemcpyHostToDevice = musaMemcpyHostToDevice;

  static constexpr auto MemcpyDeviceToHost = musaMemcpyDeviceToHost;

  static constexpr auto Memset = musaMemset;

  static constexpr auto GetDevice = [](auto &&...args) {
    return musaGetDevice(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto SetDevice = musaSetDevice;

  static constexpr auto DeviceSynchronize = [](auto &&...args) {
    return musaDeviceSynchronize(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto StreamSynchronize = musaStreamSynchronize;
};

static_assert(Runtime<Device::Type::kMoore>::Validate());

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_MOORE_RUNTIME_H_
