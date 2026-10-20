#ifndef INFINI_CCL_DEVICES_HYGON_RUNTIME_H_
#define INFINI_CCL_DEVICES_HYGON_RUNTIME_H_

#include <utility>

// clang-format off
#include <hip/hip_runtime.h>
// clang-format on

#include "devices/cuda/runtime_.h"
#include "devices/hygon/device_.h"
#include "logging.h"
#include "return_status_impl.h"

namespace infini::ccl {

template <>
struct Runtime<Device::Type::kHygon>
    : CudaRuntime<Runtime<Device::Type::kHygon>> {
  using Stream = hipStream_t;

  static constexpr Device::Type kDeviceType = Device::Type::kHygon;

  static constexpr auto Check =
      [](auto status, ReturnStatus err_code = ReturnStatus::kSystemError) {
        if (status != hipSuccess) {
          LOG(hipGetErrorString(static_cast<hipError_t>(status)));
          return err_code;
        }
        return ReturnStatus::kSuccess;
      };

  static constexpr auto Malloc = [](auto &&...args) {
    return hipMalloc(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Memcpy = hipMemcpy;

  static constexpr auto Free = hipFree;

  static constexpr auto MemcpyHostToDevice = hipMemcpyHostToDevice;

  static constexpr auto MemcpyDeviceToHost = hipMemcpyDeviceToHost;

  static constexpr auto Memset = hipMemset;

  static constexpr auto GetDevice = hipGetDevice;

  static constexpr auto SetDevice = hipSetDevice;

  static constexpr auto DeviceSynchronize = hipDeviceSynchronize;

  static constexpr auto StreamSynchronize = hipStreamSynchronize;
};

static_assert(Runtime<Device::Type::kHygon>::Validate());

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_HYGON_RUNTIME_H_
