#ifndef __INFINIOP_MATMUL_CPU_API_H__
#define __INFINIOP_MATMUL_CPU_API_H__

#include "../../../devices/cpu/cpu_handle.h"
#include "infiniop/operator.h"

struct MatmulCpuDescriptor;

typedef struct MatmulCpuDescriptor *infiniopMatmulCpuDescriptor_t;

infiniopStatus_t cpuCreateMatmulDescriptor(
    infiniopCpuHandle_t handle, infiniopMatmulCpuDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc, infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc);

infiniopStatus_t cpuGetMatmulWorkspaceSize(infiniopMatmulCpuDescriptor_t desc,
                                           size_t *size);

infiniopStatus_t cpuMatmul(infiniopMatmulCpuDescriptor_t desc, void *workspace,
                           size_t workspace_size, void *c, void const *a,
                           void const *b, float alpha, float beta);

infiniopStatus_t cpuDestroyMatmulDescriptor(infiniopMatmulCpuDescriptor_t desc);

#endif // __INFINIOP_MATMUL_CPU_API_H__
