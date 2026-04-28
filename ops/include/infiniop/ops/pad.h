#ifndef __INFINIOP_PAD_API_H__
#define __INFINIOP_PAD_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopPadDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreatePadDescriptor(infiniopHandle_t handle,
                                                               infiniopPadDescriptor_t *desc_ptr,
                                                               infiniopTensorDescriptor_t y,
                                                               infiniopTensorDescriptor_t x,
                                                               void *pad,
                                                               size_t pad_size,
                                                               const char *mode,
                                                               double value);

__INFINI_C __export infiniStatus_t infiniopGetPadWorkspaceSize(infiniopPadDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopPad(infiniopPadDescriptor_t desc,
                                               void *workspace,
                                               size_t workspace_size,
                                               void *y,
                                               const void *x,
                                               void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyPadDescriptor(infiniopPadDescriptor_t desc);

#endif
