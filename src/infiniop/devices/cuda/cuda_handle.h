#ifndef __INFINIOP_CUDA_HANDLE_H__
#define __INFINIOP_CUDA_HANDLE_H__

#include "infiniop/handle.h"

struct InfiniopCudaHandle;
typedef struct InfiniopCudaHandle *infiniopCudaHandle_t;

infiniopStatus_t createCudaHandle(infiniopCudaHandle_t *handle_ptr, int device_id, infiniDevice_t cuda_device_type);

infiniopStatus_t destroyCudaHandle(infiniopCudaHandle_t handle_ptr);

#endif
