#ifndef INFINI_OPS_TORCH_MOORE_C10_H_
#define INFINI_OPS_TORCH_MOORE_C10_H_

#include <c10/core/StreamGuard.h>
#include <c10/musa/MUSAMacros.h>
#include <c10/musa/MUSAStream.h>
#include <musa_runtime_api.h>

#include "torch/c10.h"

namespace infini::ops {

template <>
struct C10<Device::Type::kMoore> {
  static constexpr Device::Type kDeviceType = Device::Type::kMoore;

  using StreamGuard = c10::StreamGuard;

  static c10::musa::MUSAStream GetStreamFromExternal(void* stream,
                                                     int device_index) {
    return c10::musa::getStreamFromExternal(
        reinterpret_cast<musaStream_t>(stream),
        static_cast<c10::DeviceIndex>(device_index));
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_MOORE_C10_H_
