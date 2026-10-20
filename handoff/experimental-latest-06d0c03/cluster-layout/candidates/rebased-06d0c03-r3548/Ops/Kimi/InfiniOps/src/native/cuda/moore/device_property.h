#ifndef INFINI_OPS_MOORE_DEVICE_PROPERTY_H_
#define INFINI_OPS_MOORE_DEVICE_PROPERTY_H_

#include <musa_runtime.h>

namespace infini::ops {

inline int QueryMaxThreadsPerBlock() {
  int device = 0;
  musaGetDevice(&device);
  musaDeviceProp prop;
  musaGetDeviceProperties(&prop, device);
  return prop.maxThreadsPerBlock;
}

}  // namespace infini::ops

#endif
