#ifndef __INFINIOP_SWIGLU_API_H__
#define __INFINIOP_SWIGLU_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopSwiGLUDescriptor_t;

__C __export infiniStatus_t infiniopCreateSwiGLUDescriptor(infiniopHandle_t handle,
                                                           infiniopSwiGLUDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t c_desc,
                                                           infiniopTensorDescriptor_t a_desc,
                                                           infiniopTensorDescriptor_t b_desc);

__C __export infiniStatus_t infiniopGetSwiGLUWorkspaceSize(infiniopSwiGLUDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopSwiGLU(infiniopSwiGLUDescriptor_t desc,
                                           void *workspace,
                                           size_t workspace_size,
                                           void *c,
                                           void const *a,
                                           void const *b,
                                           void *stream);

__C __export infiniStatus_t infiniopDestroySwiGLUDescriptor(infiniopSwiGLUDescriptor_t desc);

#endif
