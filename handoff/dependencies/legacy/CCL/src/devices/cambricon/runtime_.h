#ifndef INFINI_CCL_DEVICES_CAMBRICON_RUNTIME_H_
#define INFINI_CCL_DEVICES_CAMBRICON_RUNTIME_H_

#include <utility>

// clang-format off
#include <cnrt.h>
// clang-format on

#include "devices/cambricon/device_.h"
#include "logging.h"
#include "return_status_impl.h"

namespace infini::ccl {

template <>
struct Runtime<Device::Type::kCambricon>
    : DeviceRuntime<Runtime<Device::Type::kCambricon>> {
  using Stream = cnrtQueue_t;

  static constexpr Device::Type kDeviceType = Device::Type::kCambricon;

  static constexpr auto Check =
      [](auto status, ReturnStatus err_code = ReturnStatus::kSystemError) {
        if (status != cnrtSuccess) {
          LOG(cnrtGetErrorStr(static_cast<cnrtRet_t>(status)));
          return err_code;
        }
        return ReturnStatus::kSuccess;
      };

  static constexpr auto Malloc = [](auto &&...args) {
    return cnrtMalloc(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Memcpy = [](auto dst, auto src, auto &&...args) {
    return cnrtMemcpy(dst, (void *)src, std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Free = cnrtFree;

  static constexpr auto MemcpyHostToDevice = cnrtMemcpyHostToDev;

  static constexpr auto MemcpyDeviceToHost = cnrtMemcpyDevToHost;

  static constexpr auto Memset = cnrtMemset;

  static constexpr auto GetDevice = cnrtGetDevice;

  static constexpr auto SetDevice = cnrtSetDevice;

  static constexpr auto DeviceSynchronize = cnrtSyncDevice;

  static constexpr auto StreamSynchronize = cnrtQueueSync;
};

static_assert(Runtime<Device::Type::kCambricon>::Validate());

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_CAMBRICON_RUNTIME_H_
