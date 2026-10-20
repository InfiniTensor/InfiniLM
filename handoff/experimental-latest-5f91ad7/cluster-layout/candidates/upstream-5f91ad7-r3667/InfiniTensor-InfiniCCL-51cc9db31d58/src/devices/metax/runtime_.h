#ifndef INFINI_CCL_DEVICES_METAX_RUNTIME_H_
#define INFINI_CCL_DEVICES_METAX_RUNTIME_H_

// clang-format off
#include <mcr/mc_runtime.h>
// clang-format on

#include "devices/cuda/runtime_.h"
#include "devices/metax/device_.h"
#include "logging.h"
#include "return_status_impl.h"

namespace infini::ccl {

template <>
struct Runtime<Device::Type::kMetax>
    : CudaRuntime<Runtime<Device::Type::kMetax>> {
  using Stream = mcStream_t;

  static constexpr Device::Type kDeviceType = Device::Type::kMetax;

  static constexpr auto Check =
      [](auto status, ReturnStatus err_code = ReturnStatus::kSystemError) {
        if (status != mcSuccess) {
          LOG(mcGetErrorString(static_cast<mcError_t>(status)));
          return err_code;
        }
        return ReturnStatus::kSuccess;
      };

  static constexpr auto Malloc = [](auto &&...args) {
    return mcMalloc(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Memcpy = mcMemcpy;

  static constexpr auto Free = mcFree;

  static constexpr auto MemcpyHostToDevice = mcMemcpyHostToDevice;

  static constexpr auto MemcpyDeviceToHost = mcMemcpyDeviceToHost;

  static constexpr auto Memset = mcMemset;

  static constexpr auto GetDevice = mcGetDevice;

  static constexpr auto SetDevice = mcSetDevice;

  static constexpr auto DeviceSynchronize = mcDeviceSynchronize;

  static constexpr auto StreamSynchronize = mcStreamSynchronize;
};

static_assert(Runtime<Device::Type::kMetax>::Validate());

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_METAX_RUNTIME_H_
