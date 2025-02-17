#ifndef __INFINIOP_SWIGLU_CPU_API_H__
#define __INFINIOP_SWIGLU_CPU_API_H__

#include "../../../devices/cpu/cpu_handle.h"
#include "infiniop/operator.h"

struct SwiGLUCpuDescriptor;

typedef struct SwiGLUCpuDescriptor *infiniopSwiGLUCpuDescriptor_t;

infiniopStatus_t cpuCreateSwiGLUDescriptor(
    infiniopCpuHandle_t handle,
    infiniopSwiGLUCpuDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc);

infiniopStatus_t cpuSwiGLU(
    infiniopSwiGLUCpuDescriptor_t desc,
    void *c, void const *a, void const *b);

infiniopStatus_t cpuDestroySwiGLUDescriptor(
    infiniopSwiGLUCpuDescriptor_t desc);

#endif // __INFINIOP_SWIGLU_CPU_API_H__
