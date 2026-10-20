#ifndef INFINI_OPS_HYGON_RUNTIME_UTILS_H_
#define INFINI_OPS_HYGON_RUNTIME_UTILS_H_

#include "native/cuda/hygon/device_property.h"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

template <>
struct RuntimeUtils<Device::Type::kHygon>
    : CudaRuntimeUtils<QueryMaxThreadsPerBlock> {
  static int GetOptimalBlockSize() {
    const int block_size =
        CudaRuntimeUtils<QueryMaxThreadsPerBlock>::GetOptimalBlockSize();
    return block_size > 256 ? 256 : block_size;
  }
};

}  // namespace infini::ops

#endif
