#ifndef INFINI_OPS_THEAD_RUNTIME_UTILS_H_
#define INFINI_OPS_THEAD_RUNTIME_UTILS_H_

#include "native/cuda/runtime_utils.h"
#include "native/cuda/thead/device_property.h"

namespace infini::ops {

template <>
struct RuntimeUtils<Device::Type::kThead>
    : CudaRuntimeUtils<QueryMaxThreadsPerBlock> {};

}  // namespace infini::ops

#endif
