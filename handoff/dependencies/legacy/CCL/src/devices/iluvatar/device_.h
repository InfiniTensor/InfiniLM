#ifndef INFINI_CCL_DEVICES_ILUVATAR_DEVICE_H_
#define INFINI_CCL_DEVICES_ILUVATAR_DEVICE_H_

#include <cuda_runtime.h>

#include "device.h"
#include "devices/cuda/checks.h"

namespace infini::ccl {

template <>
struct DeviceEnabled<Device::Type::kIluvatar> : std::true_type {};

template <>
MemorySpace GetMemorySpace<Device::Type::kIluvatar>(const void* ptr) {
  if (!ptr) {
    return MemorySpace::kHost;
  }

  cudaPointerAttributes attr;
  INFINI_CHECK_CUDA(cudaPointerGetAttributes(&attr, ptr));

  if (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged) {
    return MemorySpace::kDevice;
  }

  return MemorySpace::kHost;
}

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_ILUVATAR_DEVICE_H_
