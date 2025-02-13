#ifndef __INFINIOP_MATMUL_CPU_H__
#define __INFINIOP_MATMUL_CPU_H__

#include "../blas.h"
#include "./matmul_cpu_api.h"

typedef struct MatmulCpuDescriptor {
    infiniDevice_t device;
    infiniDtype_t dtype;
    MatmulInfo info;
} MatmulCpuDescriptor;

#endif // __INFINIOP_MATMUL_CPU_H__
