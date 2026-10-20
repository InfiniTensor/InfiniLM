#ifndef INFINI_OPS_NVIDIA_RUNTIME_UTILS_H_
#define INFINI_OPS_NVIDIA_RUNTIME_UTILS_H_

#include "native/cuda/nvidia/device_property.h"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

template <>
struct RuntimeUtils<Device::Type::kNvidia>
    : CudaRuntimeUtils<QueryMaxThreadsPerBlock> {};

}  // namespace infini::ops

#endif
