#ifndef INFINI_OPS_METAX_RUNTIME_UTILS_H_
#define INFINI_OPS_METAX_RUNTIME_UTILS_H_

#include "native/cuda/metax/device_property.h"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

template <>
struct RuntimeUtils<Device::Type::kMetax>
    : CudaRuntimeUtils<QueryMaxThreadsPerBlock> {};

}  // namespace infini::ops

#endif
