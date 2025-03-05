#ifndef __INFINIOP_CPU_HANDLE_H__
#define __INFINIOP_CPU_HANDLE_H__

#include "../../handle.h"

typedef infiniopHandle_t infiniopCpuHandle_t;

infiniStatus_t createCpuHandle(infiniopCpuHandle_t *handle_ptr);

infiniStatus_t destroyCpuHandle(infiniopCpuHandle_t handle);

#endif
