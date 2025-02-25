#ifndef __INFINIOP_CUDA_HANDLE_H__
#define __INFINIOP_CUDA_HANDLE_H__

#include "infiniop/handle.h"

struct InfiniopCudaHandle;
typedef struct InfiniopCudaHandle *infiniopCudaHandle_t;

infiniStatus_t createCudaHandle(infiniopCudaHandle_t *handle_ptr, infiniDevice_t cuda_device_type);

infiniStatus_t destroyCudaHandle(infiniopCudaHandle_t handle_ptr);

#endif
