#ifndef __INFINIOP_HANDLE_API_H__
#define __INFINIOP_HANDLE_API_H__

#include "../infinicore.h"

struct InfiniopHandle;

typedef InfiniopHandle *infiniopHandle_t;

__C __export infiniStatus_t infiniopCreateHandle(infiniopHandle_t *handle_ptr, infiniDevice_t device);

__C __export infiniStatus_t infiniopDestroyHandle(infiniopHandle_t handle);

#endif
