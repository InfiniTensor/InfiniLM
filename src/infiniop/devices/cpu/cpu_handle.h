#ifndef __INFINIOP_CPU_HANDLE_H__
#define __INFINIOP_CPU_HANDLE_H__

#include "infiniop/handle.h"

typedef infiniopHandle_t infiniopCpuHandle_t;

infiniopStatus_t createCpuHandle(infiniopCpuHandle_t *handle_ptr);

infiniopStatus_t destroyCpuHandle(infiniopCpuHandle_t handle);

#endif
