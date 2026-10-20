#ifndef INFINI_OPS_TORCH_HYGON_C10_H_
#define INFINI_OPS_TORCH_HYGON_C10_H_

#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime_api.h>

#include "torch/c10.h"

namespace infini::ops {

template <>
struct C10<Device::Type::kHygon> {
  static constexpr Device::Type kDeviceType = Device::Type::kHygon;

  using StreamGuard = c10::hip::HIPStreamGuard;

  static c10::hip::HIPStream GetStreamFromExternal(void* stream,
                                                   int device_index) {
    return c10::hip::getStreamFromExternal(
        reinterpret_cast<hipStream_t>(stream),
        static_cast<c10::DeviceIndex>(device_index));
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_HYGON_C10_H_
