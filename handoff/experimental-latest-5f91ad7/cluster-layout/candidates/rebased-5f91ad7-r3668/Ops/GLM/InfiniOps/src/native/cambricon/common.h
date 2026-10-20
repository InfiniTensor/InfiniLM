#ifndef INFINI_OPS_CAMBRICON_COMMON_H_
#define INFINI_OPS_CAMBRICON_COMMON_H_

#include <cnnl.h>
#include <cnrt.h>

#include "data_type.h"
#include "device.h"

#define NRAM_MAX_SIZE (1024 * 240)

#ifdef __BANG__

namespace infini::ops::reduce {

constexpr int batch_size = 128 / sizeof(float);

__mlu_func__ void SumInternal(float* dst, float* src, int max_batch) {
  const int width = max_batch / batch_size;

  if (width >= 4) {
    __bang_sumpool(dst, src, batch_size, 1, width, 1, width, 1, 1);
    __bang_reduce_sum(dst, dst, batch_size);
  } else {
    float sum = 0.0f;
    for (int i = 0; i < max_batch; ++i) {
      sum += src[i];
    }
    dst[0] = sum;
  }
}

}  // namespace infini::ops::reduce

#endif  // __BANG__

namespace infini::ops::cnnl_utils {

inline cnnlDataType_t GetDataType(DataType dtype) {
  switch (dtype) {
    case DataType::kInt8:
      return CNNL_DTYPE_INT8;
    case DataType::kUInt8:
      return CNNL_DTYPE_UINT8;
    case DataType::kInt32:
      return CNNL_DTYPE_INT32;
    case DataType::kInt64:
      return CNNL_DTYPE_INT64;
    case DataType::kFloat16:
      return CNNL_DTYPE_HALF;
    case DataType::kFloat32:
      return CNNL_DTYPE_FLOAT;
    case DataType::kBFloat16:
      return CNNL_DTYPE_BFLOAT16;
    case DataType::kFloat64:
      return CNNL_DTYPE_DOUBLE;
    default:
      return CNNL_DTYPE_INVALID;
  }
}

}  // namespace infini::ops::cnnl_utils

namespace infini::ops::cnrt_utils {

inline void GetLaunchConfig(const Device& device, int* core_per_cluster,
                            int* cluster_count) {
  int device_id = device.index();
  cnrtDeviceGetAttribute(cluster_count, cnrtAttrClusterCount, device_id);
  cnrtDeviceGetAttribute(core_per_cluster, cnrtAttrMcorePerCluster, device_id);
}

}  // namespace infini::ops::cnrt_utils

#endif
