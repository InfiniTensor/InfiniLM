#ifndef INFINI_OPS_NVIDIA_DEVICE_PROPERTY_H_
#define INFINI_OPS_NVIDIA_DEVICE_PROPERTY_H_

#include <cuda_runtime.h>

#include <cassert>
#include <vector>

namespace infini::ops {

class DevicePropertyCache {
 public:
  static const cudaDeviceProp& GetCurrentDeviceProps() {
    int device_id = 0;
    cudaGetDevice(&device_id);
    return GetDeviceProps(device_id);
  }

  static const cudaDeviceProp& GetDeviceProps(int device_id) {
    static std::vector<cudaDeviceProp> cache = []() {
      int count = 0;
      cudaGetDeviceCount(&count);
      if (count == 0) return std::vector<cudaDeviceProp>{};
      std::vector<cudaDeviceProp> props(count);
      for (int i = 0; i < count; ++i) {
        cudaGetDeviceProperties(&props[i], i);
      }
      return props;
    }();

    assert(device_id >= 0 && device_id < static_cast<int>(cache.size()));
    return cache[device_id];
  }
};

inline int QueryMaxThreadsPerBlock() {
  return DevicePropertyCache::GetCurrentDeviceProps().maxThreadsPerBlock;
}

}  // namespace infini::ops

#endif
