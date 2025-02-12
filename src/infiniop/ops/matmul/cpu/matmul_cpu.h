#ifndef __INFINIOP_MATMUL_CPU_H__
#define __INFINIOP_MATMUL_CPU_H__

#include "../../../devices/cpu/cpu_handle.h"
#include "../blas.h"
#include "infiniop/operator.h"

typedef struct MatmulCpuDescriptor {
    infiniDevice_t device;
    infiniDtype_t dtype;
    MatmulInfo info;
} MatmulCpuDescriptor;

typedef struct MatmulCpuDescriptor *infiniopMatmulCpuDescriptor_t;

infiniopStatus_t cpuCreateMatmulDescriptor(
    infiniopCpuHandle_t handle, infiniopMatmulCpuDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc, infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc);

infiniopStatus_t cpuGetMatmulWorkspaceSize(infiniopMatmulCpuDescriptor_t desc,
                                           uint64_t *size);

infiniopStatus_t cpuMatmul(infiniopMatmulCpuDescriptor_t desc, void *workspace,
                           uint64_t workspace_size, void *c, void const *a,
                           void const *b, float alpha, float beta);

infiniopStatus_t cpuDestroyMatmulDescriptor(infiniopMatmulCpuDescriptor_t desc);

#endif // __INFINIOP_MATMUL_CPU_H__
