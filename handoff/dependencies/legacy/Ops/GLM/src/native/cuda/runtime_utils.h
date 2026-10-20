#ifndef INFINI_OPS_CUDA_RUNTIME_UTILS_H_
#define INFINI_OPS_CUDA_RUNTIME_UTILS_H_

#include "device.h"

namespace infini::ops {

template <Device::Type device_type>
struct RuntimeUtils;

template <auto QueryMaxThreadsPerBlockFn>
struct CudaRuntimeUtils {
  static int GetOptimalBlockSize() {
    int max_threads = QueryMaxThreadsPerBlockFn();
    if (max_threads >= 2048) return 2048;
    if (max_threads >= 1024) return 1024;
    if (max_threads >= 512) return 512;
    if (max_threads >= 256) return 256;
    return 128;
  }
};

}  // namespace infini::ops

#endif
