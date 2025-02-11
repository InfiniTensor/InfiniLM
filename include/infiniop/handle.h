#ifndef __INFINIOP_HANDLE__
#define __INFINIOP_HANDLE__

#include "../infinicore.h"
#include "./status.h"

typedef struct InfiniopHandle {
    infiniDevice_t device;
    int device_id;
} InfiniopHandle;

typedef InfiniopHandle *infiniopHandle_t;

__C __export infiniopStatus_t infiniopCreateHandle(infiniopHandle_t *handle_ptr, infiniDevice_t device, int device_id);

__C __export infiniopStatus_t infiniopDestroyHandle(infiniopHandle_t handle);

#endif
