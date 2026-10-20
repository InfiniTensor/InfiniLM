#ifndef INFINI_OPS_MARS_RUNTIME_UTILS_H_
#define INFINI_OPS_MARS_RUNTIME_UTILS_H_

#include "native/cuda/mars/device_property.h"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

template <>
struct RuntimeUtils<Device::Type::kMars>
    : CudaRuntimeUtils<QueryMaxThreadsPerBlock> {};

}  // namespace infini::ops

#endif
