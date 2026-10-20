#ifndef INFINI_OPS_TORCH_ASCEND_C10_H_
#define INFINI_OPS_TORCH_ASCEND_C10_H_

#include <acl/acl_rt.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "torch/c10.h"

namespace infini::ops {

template <>
struct C10<Device::Type::kAscend> {
  static constexpr Device::Type kDeviceType = Device::Type::kAscend;

  using StreamGuard = c10_npu::NPUStreamGuard;

  static c10_npu::NPUStream GetStreamFromExternal(void* stream,
                                                  int device_index) {
    if (stream == nullptr) {
      // `torch_npu` rejects null external streams, while InfiniOps uses null
      // to select the default device stream.
      return c10_npu::getDefaultNPUStream(
          static_cast<c10::DeviceIndex>(device_index));
    }

    return c10_npu::getStreamFromExternal(
        reinterpret_cast<aclrtStream>(stream),
        static_cast<c10::DeviceIndex>(device_index));
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_ASCEND_C10_H_
