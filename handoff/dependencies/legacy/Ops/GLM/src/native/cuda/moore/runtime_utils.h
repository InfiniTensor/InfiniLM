#ifndef INFINI_OPS_MOORE_RUNTIME_UTILS_H_
#define INFINI_OPS_MOORE_RUNTIME_UTILS_H_

#include "native/cuda/moore/device_property.h"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

template <>
struct RuntimeUtils<Device::Type::kMoore>
    : CudaRuntimeUtils<QueryMaxThreadsPerBlock> {};

}  // namespace infini::ops

#endif
